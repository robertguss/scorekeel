# `scorekeel`

> **Zero-Python Pure-Rust LightGBM Scorer & Exact Lundberg TreeSHAP Explainer (DX + AX)** — Tool #14 in the [Keel Data Stack](https://github.com/robertguss/keel-stack).

Train LightGBM models in Python (`booster.save_model("model.txt")`), then deploy inference + exact **TreeSHAP** explanations anywhere in a **single ~8 MB native Rust binary** — zero Python runtime, zero SciPy/NumPy/SHAP bloat, zero C++ `lib_lightgbm` shared library, and sub-millisecond latency.

## Features

- **Native LightGBM v3/v4 `.txt` Evaluator:** Numerical splits, exact LightGBM missing-value routing (`default_left` + `MissingType::{None, Zero, NaN}`), categorical bitset splits (`cat_boundaries` / `cat_threshold`), automatic `pandas_categorical` string mapping, binary sigmoid, regression, quantile, and multiclass objectives.
- **Exact Lundberg Path-Dependent TreeSHAP in Pure Rust (`--shap`):** Matches Python `shap.TreeExplainer` to `<1e-5` precision, satisfying the Shapley efficiency axiom (`expected_value + sum(phi_i) == raw_margin`), parallelized across CPU cores via `rayon`.
- **Production Post-Processing Built In:** Optional Platt calibration (`--platt slope,intercept`), anchored log-ratio un-scaling (`--anchor-col`), plain-English feature label maps (`--labels labels.toml`), and offline zero-CDN `--html` scoring reports.
- **Multi-Format I/O:** Reads **CSV, TSV, JSONL/NDJSON, and Apache Parquet**; writes CSV or JSONL.
- **Built-In HTTP Sidecar (`scorekeel serve`):** Zero-dependency HTTP/1.1 scoring server (`GET /health`, `GET /model`, `POST /score`) using ~10 MB RAM — ideal as a sidecar next to any Render/Fly/Rails/Node/Go web app.
- **Agent-First AX:** Dual-mode TTY/JSON/agent output, `scorekeel inspect`, `scorekeel bench`, and bundled `SKILL.md` (`scorekeel skill --install`).

## Quickstart

```bash
# 1. Export any trained Python LightGBM model
#    lgbm_model.booster_.save_model("retention_dfw.txt")

# 2. Inspect model structure & split-gain importance
scorekeel inspect retention_dfw.txt

# 3. Batch score CSV / JSONL / Parquet with exact TreeSHAP explanations
scorekeel score retention_dfw.txt active_students.parquet \
  --id-col student_id \
  --shap --top-k 3 \
  --out scored.jsonl \
  --html scoring_report.html

# 4. Run a lightweight HTTP scoring sidecar (zero Python required!)
scorekeel serve retention_dfw.txt --bind 127.0.0.1:8080 --shap --top-k 3

# 5. Benchmark throughput & latency
scorekeel bench retention_dfw.txt active_students.csv --shap
```

## License

Dual-licensed under MIT or Apache-2.0.
