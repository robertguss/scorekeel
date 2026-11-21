//! Zero-CDN self-contained HTML scoring & TreeSHAP summary report.

use crate::io_data::ScoredRow;
use crate::lgbm::LgbmModel;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub fn render_score_html(
    model: &LgbmModel,
    model_path: &str,
    input_path: &str,
    rows: &[ScoredRow],
    global_shap: &[(String, f64)],
    elapsed_ms: u128,
) -> String {
    let n = rows.len().max(1);
    let mean_pred = rows.iter().map(|r| r.prediction).sum::<f64>() / (n as f64);
    let mut sorted_preds: Vec<f64> = rows.iter().map(|r| r.prediction).collect();
    sorted_preds.sort_by(|a, b| a.total_cmp(b));
    let p50 = sorted_preds[sorted_preds.len() / 2];
    let p90 = sorted_preds[(sorted_preds.len() * 9 / 10).min(sorted_preds.len() - 1)];

    // Top global SHAP SVG
    let max_sh = global_shap
        .iter()
        .map(|(_, v)| *v)
        .fold(0.01_f64, f64::max);
    let mut shap_svg = String::new();
    let shown_sh = &global_shap[..global_shap.len().min(10)];
    let svg_h = 30 + shown_sh.len() * 32;
    shap_svg.push_str(&format!(
        "<svg viewBox=\"0 0 640 {svg_h}\" width=\"100%\">"
    ));
    for (idx, (feat, val)) in shown_sh.iter().enumerate() {
        let y = 20 + idx * 32;
        let w = ((val / max_sh) * 320.0).max(4.0);
        shap_svg.push_str(&format!(
            "<text x=\"0\" y=\"{}\" font-size=\"12\" fill=\"#102a43\">{}</text>",
            y + 13,
            esc(feat)
        ));
        shap_svg.push_str(&format!(
            "<rect x=\"220\" y=\"{}\" width=\"320\" height=\"15\" rx=\"4\" fill=\"#eef2f7\"/>",
            y + 2
        ));
        shap_svg.push_str(&format!(
            "<rect x=\"220\" y=\"{}\" width=\"{w:.1}\" height=\"15\" rx=\"4\" fill=\"#0284c7\"/>",
            y + 2
        ));
        shap_svg.push_str(&format!(
            "<text x=\"550\" y=\"{}\" font-size=\"11.5\" font-weight=\"700\" fill=\"#102a43\">{val:.4}</text>",
            y + 13
        ));
    }
    shap_svg.push_str("</svg>");

    let mut ranked: Vec<&ScoredRow> = rows.iter().collect();
    ranked.sort_by(|a, b| b.prediction.total_cmp(&a.prediction));
    let top_rows_html: String = ranked
        .into_iter()
        .take(30)
        .map(|r| {
            let id_s = r
                .id
                .clone()
                .unwrap_or_else(|| format!("#{}", r.row_index));
            let drv = r
                .top_drivers
                .as_ref()
                .map(|ds| {
                    ds.iter()
                        .map(|d| {
                            let col = if d.shap_value >= 0.0 {
                                "#c53030"
                            } else {
                                "#15803d"
                            };
                            format!(
                                "<span style=\"display:inline-block;background:#eef2f7;border-left:3px solid {col};padding:2px 7px;border-radius:5px;font-size:0.78rem;margin:2px 4px 2px 0\">{}: {} ({:+.3})</span>",
                                esc(&d.label),
                                esc(&d.value),
                                d.shap_value
                            )
                        })
                        .collect::<String>()
                })
                .unwrap_or_else(|| "<span style=\"color:#52667a\">(run with --shap)</span>".to_string());
            format!(
                "<tr><td><strong>{}</strong></td><td><strong>{:.4}</strong></td><td>{:.4}</td><td>{}</td></tr>",
                esc(&id_s),
                r.prediction,
                r.raw_score,
                drv
            )
        })
        .collect();

    format!(
        r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" /><title>scorekeel Scoring &amp; TreeSHAP Report</title>
<style>
  body{{margin:0;font-family:ui-sans-serif,system-ui,sans-serif;background:#f6f8fb;color:#102a43;line-height:1.5}}
  .bar{{background:#142c4c;color:#f8fafc;padding:10px 22px;font-size:0.84rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
  .wrap{{max-width:1160px;margin:0 auto;padding:22px 18px 48px}}
  .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:16px}}
  .kpi{{background:#fff;border:1px solid #d9e2ec;border-radius:12px;padding:14px 16px}}
  .kpi .k{{font-size:0.72rem;text-transform:uppercase;color:#52667a;font-weight:700}}
  .kpi .v{{font-size:1.45rem;font-weight:800;margin-top:4px}}
  .card{{background:#fff;border:1px solid #d9e2ec;border-radius:12px;padding:18px 20px;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-size:0.86rem}}
  th,td{{text-align:left;padding:8px;border-bottom:1px solid #d9e2ec;vertical-align:top}}
  th{{font-size:0.74rem;text-transform:uppercase;color:#52667a}}
</style></head><body>
<div class="bar">
  <div><strong>scorekeel v{ver}</strong> &mdash; Pure-Rust LightGBM + Exact TreeSHAP Report</div>
  <div>Model SHA-256: <code>{sha}</code> &middot; {elapsed_ms}ms</div>
</div>
<div class="wrap">
  <div class="kpis">
    <div class="kpi"><div class="k">Rows Scored</div><div class="v">{nrows}</div></div>
    <div class="kpi"><div class="k">Mean / P50 Score</div><div class="v">{mean_pred:.4} / {p50:.4}</div></div>
    <div class="kpi"><div class="k">P90 Score</div><div class="v">{p90:.4}</div></div>
    <div class="kpi"><div class="k">Model Ensemble</div><div class="v">{ntrees} trees</div></div>
  </div>
  <div class="card"><h2>Model &amp; Input</h2><p style="color:#52667a;margin:0">Model: <code>{mp}</code> ({obj}) &middot; Input: <code>{ip}</code> &middot; Features: {nf}</p></div>
  <div class="card"><h2>Global Mean |TreeSHAP| Feature Importance</h2>{shap_svg}</div>
  <div class="card"><h2>Top 30 Scored Records</h2><table><thead><tr><th>Record</th><th>Prediction</th><th>Raw Margin</th><th>Exact TreeSHAP Top Drivers</th></tr></thead><tbody>{top_rows_html}</tbody></table></div>
</div></body></html>"#,
        ver = env!("CARGO_PKG_VERSION"),
        sha = &model.sha256[..16],
        nrows = rows.len(),
        ntrees = model.trees.len(),
        mp = esc(model_path),
        obj = esc(&model.objective_raw),
        ip = esc(input_path),
        nf = model.feature_names.len(),
    )
}
