use std::io::IsTerminal;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "scorekeel",
    version,
    about = "Zero-Python pure-Rust LightGBM scorer & exact Lundberg TreeSHAP explainer (DX + AX)."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Auto,
    Human,
    Json,
    Agent,
}

impl OutputFormat {
    pub fn resolve(self) -> ResolvedFormat {
        match self {
            Self::Human => ResolvedFormat::Human,
            Self::Json => ResolvedFormat::Json,
            Self::Agent => ResolvedFormat::Agent,
            Self::Auto => {
                if std::io::stdout().is_terminal() {
                    ResolvedFormat::Human
                } else {
                    ResolvedFormat::Json
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedFormat {
    Human,
    Json,
    Agent,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Inspect a LightGBM .txt model: objective, trees, depth, features, categoricals & gain importance.
    Inspect {
        /// Path to LightGBM text model (`booster.save_model("model.txt")`).
        model: PathBuf,

        /// Output format: auto, human, json, or agent.
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Score CSV, JSONL, or Parquet rows in parallel with optional exact TreeSHAP explanations.
    Score {
        /// Path to LightGBM text model (.txt).
        model: PathBuf,

        /// Input data file (.csv, .jsonl, .ndjson, .parquet).
        input: PathBuf,

        /// Compute exact Lundberg path-dependent TreeSHAP explanations per row.
        #[arg(long, default_value_t = false)]
        shap: bool,

        /// Number of top SHAP drivers to emit per row when `--shap` is set.
        #[arg(long, default_value_t = 3)]
        top_k: usize,

        /// Optional Platt calibration `slope,intercept` applied to binary log-odds.
        #[arg(long)]
        platt: Option<String>,

        /// Optional anchor column for log-ratio regression models: `(anchor+1)*exp(raw)-1`.
        #[arg(long)]
        anchor_col: Option<String>,

        /// Optional ID column to carry into scored output & HTML report.
        #[arg(long)]
        id_col: Option<String>,

        /// Optional TOML feature-label map (`[labels]\nfeat = "Plain English"`).
        #[arg(long)]
        labels: Option<PathBuf>,

        /// Write scored rows to CSV or JSONL (inferred from extension).
        #[arg(long)]
        out: Option<PathBuf>,

        /// Write a self-contained offline HTML scoring & SHAP summary report.
        #[arg(long)]
        html: Option<PathBuf>,

        /// Output format: auto, human, json, or agent.
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Start a zero-dependency HTTP JSON scoring microservice (`GET /health`, `GET /model`, `POST /score`).
    Serve {
        /// Path to LightGBM text model (.txt).
        model: PathBuf,

        /// Bind address (default 127.0.0.1:8080).
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,

        /// Include exact TreeSHAP top-K drivers in every `/score` response.
        #[arg(long, default_value_t = false)]
        shap: bool,

        /// Number of top SHAP drivers per row when `--shap` is enabled.
        #[arg(long, default_value_t = 3)]
        top_k: usize,

        /// Optional Platt calibration `slope,intercept`.
        #[arg(long)]
        platt: Option<String>,

        /// Optional anchor column for log-ratio regression models.
        #[arg(long)]
        anchor_col: Option<String>,
    },

    /// Benchmark pure-Rust inference throughput (rows/sec & p50/p99 latency).
    Bench {
        /// Path to LightGBM text model (.txt).
        model: PathBuf,

        /// Input data file (.csv, .jsonl, .parquet).
        input: PathBuf,

        /// Also benchmark exact Lundberg TreeSHAP computation.
        #[arg(long, default_value_t = false)]
        shap: bool,

        /// Number of benchmark iterations over the input file.
        #[arg(long, default_value_t = 5)]
        iters: usize,

        /// Output format: auto, human, json, or agent.
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Print or install the bundled SKILL.md manual for AI coding agents.
    Skill {
        /// Install SKILL.md into `<DIR>/scorekeel/SKILL.md` (default `.agents/skills`).
        #[arg(long, num_args = 0..=1, default_missing_value = ".agents/skills")]
        install: Option<PathBuf>,

        /// Output format (accepted for Keel Stack CLI parity).
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },
}
