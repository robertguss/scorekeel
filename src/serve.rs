//! Lightweight zero-dependency HTTP/1.1 scoring microservice (`scorekeel serve`).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use rayon::prelude::*;

use crate::io_data::{ScoredRow, ShapDriver, json_obj_to_map};
use crate::lgbm::LgbmModel;
use crate::shap;

pub struct ServeConfig {
    pub model: Arc<LgbmModel>,
    pub compute_shap: bool,
    pub top_k: usize,
    pub platt: Option<(f64, f64)>,
    pub anchor_col: Option<String>,
}

pub fn run_server(bind: &str, cfg: ServeConfig) -> Result<(), String> {
    let listener =
        TcpListener::bind(bind).map_err(|e| format!("failed to bind {bind}: {e}"))?;
    eprintln!(
        "scorekeel HTTP sidecar listening on http://{bind} ({} trees, {} features, shap={})",
        cfg.model.trees.len(),
        cfg.model.feature_names.len(),
        cfg.compute_shap
    );
    eprintln!("  GET  /health -> service liveness");
    eprintln!("  GET  /model  -> model metadata & feature schema");
    eprintln!("  POST /score  -> score JSON object or array of objects");

    let cfg = Arc::new(cfg);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let c = Arc::clone(&cfg);
                std::thread::spawn(move || {
                    let _ = handle_conn(s, &c);
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

fn handle_conn(mut stream: TcpStream, cfg: &ServeConfig) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return write_json(&mut stream, 400, r#"{"error":"malformed request"}"#);
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_len: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
        {
            content_len = v.trim().parse().unwrap_or(0);
        }
    }

    match (method, path) {
        ("GET", "/health") => write_json(
            &mut stream,
            200,
            &serde_json::json!({
                "status": "ok",
                "tool": "scorekeel",
                "version": env!("CARGO_PKG_VERSION"),
                "trees": cfg.model.trees.len(),
            })
            .to_string(),
        ),
        ("GET", "/model") => write_json(
            &mut stream,
            200,
            &serde_json::json!({
                "sha256": cfg.model.sha256,
                "objective": cfg.model.objective_raw,
                "trees": cfg.model.trees.len(),
                "features": cfg.model.feature_names,
                "categorical_levels": cfg.model.categorical_levels,
            })
            .to_string(),
        ),
        ("POST", "/score") => {
            if content_len > 16 * 1024 * 1024 {
                return write_json(&mut stream, 413, r#"{"error":"payload exceeds 16 MiB"}"#);
            }
            let mut body = vec![0u8; content_len];
            reader.read_exact(&mut body)?;
            let parsed: Result<serde_json::Value, _> = serde_json::from_slice(&body);
            let Ok(val) = parsed else {
                return write_json(&mut stream, 400, r#"{"error":"invalid JSON body"}"#);
            };
            let records_res: Result<Vec<HashMap<String, String>>, String> =
                if let Some(arr) = val.as_array() {
                    arr.iter().map(json_obj_to_map).collect()
                } else if let Some(rows_arr) = val.get("rows").and_then(|v| v.as_array()) {
                    rows_arr.iter().map(json_obj_to_map).collect()
                } else {
                    json_obj_to_map(&val).map(|m| vec![m])
                };
            let Ok(records) = records_res else {
                return write_json(&mut stream, 400, r#"{"error":"expected JSON object or array"}"#);
            };

            let scored: Vec<ScoredRow> = records
                .par_iter()
                .enumerate()
                .map(|(idx, rec)| {
                    let x = cfg.model.encode_record(rec);
                    let raw = cfg.model.predict_raw(&x);
                    let anc = cfg
                        .anchor_col
                        .as_deref()
                        .and_then(|c| rec.get(c))
                        .and_then(|s| s.parse::<f64>().ok());
                    let pred = cfg.model.predict_transformed(&x, cfg.platt, anc);
                    let (ev, drivers) = if cfg.compute_shap {
                        let (e, phi) = shap::explain_row(&cfg.model, &x);
                        let mut idxs: Vec<usize> = (0..phi.len()).collect();
                        idxs.sort_by(|&a, &b| phi[b].abs().total_cmp(&phi[a].abs()));
                        let ds: Vec<ShapDriver> = idxs
                            .into_iter()
                            .take(cfg.top_k)
                            .filter(|&fi| phi[fi].abs() > 1e-6)
                            .map(|fi| {
                                let fname = cfg.model.feature_names[fi].clone();
                                ShapDriver {
                                    label: fname.clone(),
                                    value: rec.get(&fname).cloned().unwrap_or_else(|| "NaN".to_string()),
                                    feature: fname,
                                    shap_value: (phi[fi] * 1_000_000.0).round() / 1_000_000.0,
                                }
                            })
                            .collect();
                        (Some((e * 1_000_000.0).round() / 1_000_000.0), Some(ds))
                    } else {
                        (None, None)
                    };
                    ScoredRow {
                        row_index: idx,
                        id: rec.get("id").cloned(),
                        raw_score: (raw * 1_000_000.0).round() / 1_000_000.0,
                        prediction: (pred * 1_000_000.0).round() / 1_000_000.0,
                        expected_value: ev,
                        top_drivers: drivers,
                        shap_values: None,
                    }
                })
                .collect();

            let resp = serde_json::json!({
                "count": scored.len(),
                "predictions": scored,
            })
            .to_string();
            write_json(&mut stream, 200, &resp)
        }
        _ => write_json(&mut stream, 404, r#"{"error":"not found"}"#),
    }
}

fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}
