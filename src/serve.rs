//! Hardened zero-dependency HTTP/1.1 scoring microservice (`scorekeel serve`).
//!
//! Defends against Slowloris (socket read/write timeouts), header OOM (bounded line
//! and total header sizes), thread/FD exhaustion (fixed worker pool + bounded queue
//! returning 503), and RFC 9112 request smuggling (strict Content-Length / Transfer-Encoding).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use rayon::prelude::*;

use crate::io_data::{ScoredRow, ShapDriver, json_obj_to_map};
use crate::lgbm::LgbmModel;
use crate::shap;

const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_LINE_BYTES: usize = 8 * 1024;
const MAX_TOTAL_HEADER_BYTES: usize = 32 * 1024;
const MAX_HEADERS: usize = 64;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(5);
const QUEUE_CAPACITY: usize = 64;

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
    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get().clamp(4, 16))
        .unwrap_or(8);

    eprintln!(
        "scorekeel HTTP sidecar listening on http://{bind} ({} trees, {} features, shap={}, workers={num_workers})",
        cfg.model.trees.len(),
        cfg.model.feature_names.len(),
        cfg.compute_shap
    );
    eprintln!("  GET  /health -> service liveness");
    eprintln!("  GET  /model  -> model metadata & feature schema");
    eprintln!("  POST /score  -> score JSON object or array of objects");

    let cfg = Arc::new(cfg);
    let (tx, rx) = mpsc::sync_channel::<TcpStream>(QUEUE_CAPACITY);
    let rx = Arc::new(Mutex::new(rx));

    for idx in 0..num_workers {
        let rx_c = Arc::clone(&rx);
        let cfg_c = Arc::clone(&cfg);
        std::thread::Builder::new()
            .name(format!("scorekeel-worker-{idx}"))
            .spawn(move || loop {
                let stream = {
                    let Ok(guard) = rx_c.lock() else {
                        break;
                    };
                    match guard.recv() {
                        Ok(s) => s,
                        Err(_) => break,
                    }
                };
                let _ = handle_conn(stream, &cfg_c);
            })
            .map_err(|e| format!("failed spawning worker thread: {e}"))?;
    }

    for stream in listener.incoming() {
        match stream {
            Ok(s) => match tx.try_send(s) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(mut s)) => {
                    let _ = s.set_write_timeout(Some(Duration::from_secs(2)));
                    let _ = write_json(&mut s, 503, r#"{"error":"server busy"}"#);
                }
                Err(mpsc::TrySendError::Disconnected(_)) => break,
            },
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

enum ParsedRequest {
    Empty,
    Ready {
        method: String,
        path: String,
        body: Vec<u8>,
    },
    HttpError(u16, &'static str),
}

fn read_bounded_line(
    reader: &mut BufReader<&mut TcpStream>,
    max_bytes: usize,
) -> Result<Option<String>, Result<u16, std::io::Error>> {
    let mut buf = String::new();
    let mut limited = reader.take((max_bytes + 1) as u64);
    let n = limited.read_line(&mut buf).map_err(Err)?;
    if n == 0 {
        return Ok(None);
    }
    if buf.len() > max_bytes || (!buf.ends_with('\n') && buf.len() == max_bytes + 1) {
        return Err(Ok(431));
    }
    Ok(Some(buf))
}

fn read_http_request(stream: &mut TcpStream) -> std::io::Result<ParsedRequest> {
    let mut reader = BufReader::new(stream);
    let request_line = match read_bounded_line(&mut reader, MAX_REQUEST_LINE_BYTES) {
        Ok(Some(l)) => l,
        Ok(None) => return Ok(ParsedRequest::Empty),
        Err(Ok(_)) => {
            return Ok(ParsedRequest::HttpError(
                414,
                r#"{"error":"request line too long"}"#,
            ));
        }
        Err(Err(e)) => return Err(e),
    };

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(ParsedRequest::HttpError(
            400,
            r#"{"error":"malformed request line"}"#,
        ));
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    let mut seen_content_len: Option<usize> = None;
    let mut has_transfer_encoding = false;
    let mut header_count = 0usize;
    let mut total_header_bytes = request_line.len();

    loop {
        let line = match read_bounded_line(&mut reader, MAX_HEADER_LINE_BYTES) {
            Ok(Some(l)) => l,
            Ok(None) => {
                return Ok(ParsedRequest::HttpError(
                    400,
                    r#"{"error":"unexpected EOF in headers"}"#,
                ));
            }
            Err(Ok(code)) => {
                return Ok(ParsedRequest::HttpError(
                    code,
                    r#"{"error":"header line exceeds limit"}"#,
                ));
            }
            Err(Err(e)) => return Err(e),
        };
        if line == "\r\n" || line == "\n" {
            break;
        }
        header_count += 1;
        total_header_bytes += line.len();
        if header_count > MAX_HEADERS || total_header_bytes > MAX_TOTAL_HEADER_BYTES {
            return Ok(ParsedRequest::HttpError(
                431,
                r#"{"error":"request header fields too large"}"#,
            ));
        }
        let Some((k, v)) = line.split_once(':') else {
            return Ok(ParsedRequest::HttpError(
                400,
                r#"{"error":"malformed header"}"#,
            ));
        };
        let key = k.trim();
        let val = v.trim();
        if key.eq_ignore_ascii_case("transfer-encoding") {
            has_transfer_encoding = true;
        } else if key.eq_ignore_ascii_case("content-length") {
            if val.is_empty() || !val.bytes().all(|b| b.is_ascii_digit()) {
                return Ok(ParsedRequest::HttpError(
                    400,
                    r#"{"error":"invalid Content-Length"}"#,
                ));
            }
            let Ok(n) = val.parse::<usize>() else {
                return Ok(ParsedRequest::HttpError(
                    400,
                    r#"{"error":"Content-Length out of range"}"#,
                ));
            };
            if let Some(prev) = seen_content_len
                && prev != n
            {
                return Ok(ParsedRequest::HttpError(
                    400,
                    r#"{"error":"conflicting duplicate Content-Length headers"}"#,
                ));
            }
            seen_content_len = Some(n);
        }
    }

    if has_transfer_encoding {
        if seen_content_len.is_some() {
            return Ok(ParsedRequest::HttpError(
                400,
                r#"{"error":"both Transfer-Encoding and Content-Length present"}"#,
            ));
        }
        return Ok(ParsedRequest::HttpError(
            501,
            r#"{"error":"Transfer-Encoding is not supported; send Content-Length"}"#,
        ));
    }

    let content_len = seen_content_len.unwrap_or(0);
    if content_len > MAX_BODY_BYTES {
        return Ok(ParsedRequest::HttpError(
            413,
            r#"{"error":"payload exceeds 16 MiB"}"#,
        ));
    }

    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader.read_exact(&mut body)?;
    }

    Ok(ParsedRequest::Ready { method, path, body })
}

fn handle_conn(mut stream: TcpStream, cfg: &ServeConfig) -> std::io::Result<()> {
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;

    let req = match read_http_request(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            if matches!(
                e.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) {
                let _ = write_json(&mut stream, 408, r#"{"error":"request timeout"}"#);
            }
            return Ok(());
        }
    };

    let (method, path, body) = match req {
        ParsedRequest::Empty => return Ok(()),
        ParsedRequest::HttpError(code, msg) => return write_json(&mut stream, code, msg),
        ParsedRequest::Ready { method, path, body } => (method, path, body),
    };

    match (method.as_str(), path.as_str()) {
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
                return write_json(
                    &mut stream,
                    400,
                    r#"{"error":"expected JSON object or array"}"#,
                );
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
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|v| v.is_finite());
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
                                    value: rec
                                        .get(&fname)
                                        .cloned()
                                        .unwrap_or_else(|| "NaN".to_string()),
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
        408 => "Request Timeout",
        413 => "Payload Too Large",
        414 => "URI Too Long",
        431 => "Request Header Fields Too Large",
        501 => "Not Implemented",
        503 => "Service Unavailable",
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
