---
name: scorekeel
description: Zero-Python single-binary Rust LightGBM scorer & exact Lundberg TreeSHAP explainer for CSV, JSONL, Parquet, and lightweight HTTP scoring sidecars.
---

# `scorekeel` — Pure-Rust LightGBM + Exact TreeSHAP Scorer

## When to Use
- Deploy Python-trained LightGBM models (`booster.save_model("model.txt")`) in production without shipping Python, SciPy, NumPy, or `lib_lightgbm`.
- Score CSV, JSONL, or Parquet files in parallel (`rayon`) with exact Lundberg TreeSHAP feature attributions (`--shap --top-k 3`).
- Run a sub-millisecond HTTP JSON scoring sidecar (`scorekeel serve model.txt --bind 127.0.0.1:8080 --shap`) next to any web app.
- Inspect LightGBM `.txt` models (`scorekeel inspect model.txt --format agent`) for feature names, categorical levels, tree depth, and global gain importance.

## Command Matrix

| Task | Command |
| --- | --- |
| Inspect a LightGBM `.txt` model | `scorekeel inspect model.txt --format agent` |
| Score CSV/JSONL/Parquet + exact TreeSHAP | `scorekeel score model.txt data.csv --shap --top-k 3 --out scored.jsonl` |
| Score with Platt calibration + HTML report | `scorekeel score model.txt data.parquet --platt 1.05,-0.12 --shap --html report.html` |
| Score anchored log-ratio demand model | `scorekeel score model.txt courses.csv --anchor-col same_season_last_year --shap` |
| Run lightweight HTTP scoring server | `scorekeel serve model.txt --bind 127.0.0.1:8080 --shap` |
| Benchmark throughput & p99 latency | `scorekeel bench model.txt data.csv --shap` |
| Print/install this skill | `scorekeel skill` / `scorekeel skill --install` |

## Exit Codes
- `0`: Success.
- `1`: Validation or scoring threshold check failed.
- `2`: I/O, model parse, or malformed input error.
