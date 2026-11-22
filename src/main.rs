//! `scorekeel` — Zero-Python pure-Rust LightGBM scorer & exact TreeSHAP explainer.

mod cli;
mod html;
mod io_data;
mod lgbm;
mod serve;
mod shap;
mod skill;

use std::collections::HashMap;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use cli::{Cli, Command, ResolvedFormat};
use io_data::{ScoredRow, ShapDriver, load_label_map, read_records, write_scored_output};
use lgbm::LgbmModel;
use owo_colors::OwoColorize;
use rayon::prelude::*;

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}

fn parse_platt(raw: Option<&str>) -> Result<Option<(f64, f64)>, String> {
    let Some(s) = raw else {
        return Ok(None);
    };
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("--platt expects `slope,intercept`, got `{s}`"));
    }
    let slope: f64 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("invalid platt slope `{}`", parts[0]))?;
    let intercept: f64 = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("invalid platt intercept `{}`", parts[1]))?;
    if !slope.is_finite() || !intercept.is_finite() {
        return Err(format!(
            "--platt slope and intercept must be finite numbers, got `{s}`"
        ));
    }
    Ok(Some((slope, intercept)))
}

fn dispatch(cli: Cli) -> Result<u8, String> {
    match cli.command {
        Command::Skill { install, format: _ } => {
            skill::handle_skill(install.as_deref())?;
            Ok(0)
        }
        Command::Inspect { model, format } => {
            let m = LgbmModel::from_file(&model)?;
            let max_depth = m.trees.iter().map(|t| t.max_depth()).max().unwrap_or(0);
            let avg_leaves = m.trees.iter().map(|t| t.num_leaves as f64).sum::<f64>()
                / (m.trees.len().max(1) as f64);
            let gains = m.gain_importance();
            let payload = serde_json::json!({
                "tool": "scorekeel",
                "version": env!("CARGO_PKG_VERSION"),
                "model_path": model.display().to_string(),
                "sha256": m.sha256,
                "lgbm_version": m.version,
                "objective": m.objective_raw,
                "num_trees": m.trees.len(),
                "num_class": m.num_class,
                "num_features": m.feature_names.len(),
                "max_depth": max_depth,
                "avg_leaves": (avg_leaves * 100.0).round() / 100.0,
                "features": m.feature_names,
                "categorical_levels": m.categorical_levels,
                "top_gain_features": gains.iter().take(12).map(|(f, g, sp)| serde_json::json!({"feature": f, "gain": g, "splits": sp})).collect::<Vec<_>>(),
            });
            match format.resolve() {
                ResolvedFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
                }
                ResolvedFormat::Agent => {
                    println!("{}", serde_json::to_string(&payload).unwrap());
                }
                ResolvedFormat::Human => {
                    println!(
                        "{} inspect {} | sha256: {}",
                        "scorekeel".bold(),
                        model.display(),
                        &m.sha256[..12]
                    );
                    println!(
                        "  objective: {} | trees: {} | features: {} | max_depth: {} | avg_leaves: {:.1}",
                        m.objective_raw,
                        m.trees.len(),
                        m.feature_names.len(),
                        max_depth,
                        avg_leaves
                    );
                    if !m.categorical_levels.is_empty() {
                        println!(
                            "  categoricals: {}",
                            m.categorical_levels
                                .iter()
                                .map(|(k, v)| format!("{k} ({} levels)", v.len()))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                    println!("\n{}", "Top Features by Split Gain:".bold());
                    for (feat, gain, splits) in gains.iter().take(10) {
                        println!("  {:<32} gain={:>10.2}  splits={}", feat, gain, splits);
                    }
                }
            }
            Ok(0)
        }
        Command::Score {
            model,
            input,
            shap: compute_shap,
            top_k,
            platt,
            anchor_col,
            id_col,
            labels,
            out,
            html: html_path,
            format,
        } => {
            let t0 = Instant::now();
            let m = LgbmModel::from_file(&model)?;
            let platt_params = parse_platt(platt.as_deref())?;
            let label_map = load_label_map(labels.as_deref())?;
            let records = read_records(&input)?;

            let scored: Vec<ScoredRow> = records
                .par_iter()
                .enumerate()
                .map(|(idx, rec)| {
                    let x = m.encode_record(rec);
                    let raw = m.predict_raw(&x);
                    let anc = anchor_col
                        .as_deref()
                        .and_then(|c| rec.get(c))
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|v| v.is_finite());
                    let pred = m.predict_transformed(&x, platt_params, anc);
                    let id = id_col.as_deref().and_then(|c| rec.get(c).cloned());

                    let (ev, top_drivers, shap_map) = if compute_shap {
                        let (e, phi) = shap::explain_row(&m, &x);
                        let mut idxs: Vec<usize> = (0..phi.len()).collect();
                        idxs.sort_by(|&a, &b| phi[b].abs().total_cmp(&phi[a].abs()));
                        let drivers: Vec<ShapDriver> = idxs
                            .into_iter()
                            .take(top_k)
                            .filter(|&fi| phi[fi].abs() > 1e-6)
                            .map(|fi| {
                                let fname = m.feature_names[fi].clone();
                                let lbl = label_map
                                    .get(&fname)
                                    .cloned()
                                    .unwrap_or_else(|| fname.clone());
                                let val_str = rec
                                    .get(&fname)
                                    .cloned()
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or_else(|| "NaN".to_string());
                                ShapDriver {
                                    feature: fname,
                                    label: lbl,
                                    value: val_str,
                                    shap_value: (phi[fi] * 1_000_000.0).round() / 1_000_000.0,
                                }
                            })
                            .collect();
                        let full_map: HashMap<String, f64> = m
                            .feature_names
                            .iter()
                            .enumerate()
                            .map(|(fi, fnm)| {
                                (fnm.clone(), (phi[fi] * 1_000_000.0).round() / 1_000_000.0)
                            })
                            .collect();
                        (
                            Some((e * 1_000_000.0).round() / 1_000_000.0),
                            Some(drivers),
                            Some(full_map),
                        )
                    } else {
                        (None, None, None)
                    };

                    ScoredRow {
                        row_index: idx,
                        id,
                        raw_score: (raw * 1_000_000.0).round() / 1_000_000.0,
                        prediction: (pred * 1_000_000.0).round() / 1_000_000.0,
                        expected_value: ev,
                        top_drivers,
                        shap_values: shap_map,
                    }
                })
                .collect();

            let elapsed_ms = t0.elapsed().as_millis();

            // Compute global mean |SHAP|
            let mut global_shap: Vec<(String, f64)> = Vec::new();
            if compute_shap && !scored.is_empty() {
                let mut sums = vec![0.0_f64; m.feature_names.len()];
                for r in &scored {
                    if let Some(smap) = &r.shap_values {
                        for (fi, fname) in m.feature_names.iter().enumerate() {
                            sums[fi] += smap.get(fname).copied().unwrap_or(0.0).abs();
                        }
                    }
                }
                let n = scored.len() as f64;
                global_shap = m
                    .feature_names
                    .iter()
                    .enumerate()
                    .map(|(fi, f)| {
                        let lbl = label_map.get(f).cloned().unwrap_or_else(|| f.clone());
                        (lbl, (sums[fi] / n * 10000.0).round() / 10000.0)
                    })
                    .collect();
                global_shap.sort_by(|a, b| b.1.total_cmp(&a.1));
            }

            if let Some(out_path) = &out {
                write_scored_output(out_path, &scored)?;
                eprintln!(
                    "wrote {} scored rows -> {}",
                    scored.len(),
                    out_path.display()
                );
            }
            if let Some(hp) = &html_path {
                let rendered = html::render_score_html(
                    &m,
                    &model.display().to_string(),
                    &input.display().to_string(),
                    &scored,
                    &global_shap,
                    elapsed_ms,
                );
                if let Some(parent) = hp.parent()
                    && !parent.as_os_str().is_empty()
                {
                    fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
                }
                fs::write(hp, rendered)
                    .map_err(|e| format!("failed writing {}: {e}", hp.display()))?;
                eprintln!("wrote HTML report -> {}", hp.display());
            }

            let mean_pred = if scored.is_empty() {
                0.0
            } else {
                scored.iter().map(|r| r.prediction).sum::<f64>() / (scored.len() as f64)
            };
            let summary = serde_json::json!({
                "tool": "scorekeel",
                "version": env!("CARGO_PKG_VERSION"),
                "model_sha256": m.sha256,
                "rows_scored": scored.len(),
                "elapsed_ms": elapsed_ms,
                "mean_prediction": (mean_pred * 1_000_000.0).round() / 1_000_000.0,
                "shap_enabled": compute_shap,
                "global_shap_top": global_shap.iter().take(8).map(|(f, v)| serde_json::json!({"feature": f, "mean_abs_shap": v})).collect::<Vec<_>>(),
                "predictions": if out.is_some() { serde_json::json!(scored.iter().take(5).collect::<Vec<_>>()) } else { serde_json::json!(scored) },
            });

            match format.resolve() {
                ResolvedFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
                }
                ResolvedFormat::Agent => {
                    println!("{}", serde_json::to_string(&summary).unwrap());
                }
                ResolvedFormat::Human => {
                    println!(
                        "{} scored {} rows in {}ms (mean={:.4}, shap={})",
                        "scorekeel".bold().green(),
                        scored.len(),
                        elapsed_ms,
                        mean_pred,
                        compute_shap
                    );
                    for r in scored.iter().take(10) {
                        let id_s = r.id.clone().unwrap_or_else(|| format!("#{}", r.row_index));
                        let drv = r
                            .top_drivers
                            .as_ref()
                            .map(|ds| {
                                ds.iter()
                                    .map(|d| {
                                        format!("{}={} ({:+.3})", d.label, d.value, d.shap_value)
                                    })
                                    .collect::<Vec<_>>()
                                    .join(" | ")
                            })
                            .unwrap_or_default();
                        println!(
                            "  {:<14} pred={:.4}  raw={:+.4}  {}",
                            id_s, r.prediction, r.raw_score, drv
                        );
                    }
                }
            }
            Ok(0)
        }
        Command::Serve {
            model,
            bind,
            shap: compute_shap,
            top_k,
            platt,
            anchor_col,
        } => {
            let m = Arc::new(LgbmModel::from_file(&model)?);
            let platt_params = parse_platt(platt.as_deref())?;
            serve::run_server(
                &bind,
                serve::ServeConfig {
                    model: m,
                    compute_shap,
                    top_k,
                    platt: platt_params,
                    anchor_col,
                },
            )?;
            Ok(0)
        }
        Command::Bench {
            model,
            input,
            shap: compute_shap,
            iters,
            format,
        } => {
            let m = LgbmModel::from_file(&model)?;
            let records = read_records(&input)?;
            let encoded: Vec<Vec<f64>> = records.iter().map(|r| m.encode_record(r)).collect();
            let n = encoded.len().max(1);
            let mut latencies_us = Vec::with_capacity(iters);
            let mut checksum = 0.0_f64;

            for _ in 0..iters.max(1) {
                let t0 = Instant::now();
                let s: f64 = encoded
                    .par_iter()
                    .map(|x| {
                        let p = m.predict_raw(x);
                        if compute_shap {
                            let (_, phi) = shap::explain_row(&m, x);
                            p + phi.iter().sum::<f64>()
                        } else {
                            p
                        }
                    })
                    .sum();
                checksum += s;
                latencies_us.push(t0.elapsed().as_micros());
            }
            latencies_us.sort_unstable();
            let p50_us = latencies_us[latencies_us.len() / 2];
            let p99_us = latencies_us[(latencies_us.len() * 99 / 100).min(latencies_us.len() - 1)];
            let rows_per_sec = (n as f64) / ((p50_us.max(1) as f64) / 1_000_000.0);

            let out = serde_json::json!({
                "tool": "scorekeel",
                "rows_per_iter": n,
                "iters": iters.max(1),
                "trees": m.trees.len(),
                "shap": compute_shap,
                "p50_ms": (p50_us as f64) / 1000.0,
                "p99_ms": (p99_us as f64) / 1000.0,
                "us_per_row": ((p50_us as f64) / (n as f64) * 100.0).round() / 100.0,
                "rows_per_sec": rows_per_sec.round(),
                "checksum": (checksum * 1000.0).round() / 1000.0,
            });
            match format.resolve() {
                ResolvedFormat::Json => println!("{}", serde_json::to_string_pretty(&out).unwrap()),
                ResolvedFormat::Agent => println!("{}", serde_json::to_string(&out).unwrap()),
                ResolvedFormat::Human => {
                    println!(
                        "{} bench ({} trees, {} rows, shap={}): p50={:.2}ms | {:.1} µs/row | {:.0} rows/sec",
                        "scorekeel".bold().green(),
                        m.trees.len(),
                        n,
                        compute_shap,
                        (p50_us as f64) / 1000.0,
                        (p50_us as f64) / (n as f64),
                        rows_per_sec
                    );
                }
            }
            Ok(0)
        }
    }
}
