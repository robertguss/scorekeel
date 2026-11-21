//! Native pure-Rust LightGBM text booster parser & evaluator.
//!
//! Supports LightGBM v3/v4 text format (`booster.save_model("model.txt")`),
//! including numerical thresholds, missing-value routing (`default_left` +
//! `MissingType::{None, Zero, NaN}`), categorical bitset splits
//! (`cat_boundaries` + `cat_threshold`), `pandas_categorical` string lookups,
//! binary sigmoid, regression, quantile, and multiclass objectives.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    BinarySigmoid,
    Regression,
    Quantile,
    Multiclass(usize),
    Other(String),
}

#[derive(Debug, Clone)]
pub struct Tree {
    pub num_leaves: usize,
    pub split_feature: Vec<usize>,
    pub split_gain: Vec<f64>,
    pub threshold: Vec<f64>,
    pub decision_type: Vec<u8>,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub leaf_value: Vec<f64>,
    pub leaf_count: Vec<f64>,
    pub internal_count: Vec<f64>,
    pub cat_boundaries: Vec<usize>,
    pub cat_threshold: Vec<u32>,
}

impl Tree {
    /// Evaluate a single decision tree on feature vector `x`.
    #[inline]
    pub fn predict_row(&self, x: &[f64]) -> f64 {
        if self.num_leaves <= 1 || self.split_feature.is_empty() {
            return self.leaf_value.first().copied().unwrap_or(0.0);
        }
        let mut node: i32 = 0;
        while node >= 0 {
            let idx = node as usize;
            let feat = self.split_feature[idx];
            let fval = x.get(feat).copied().unwrap_or(f64::NAN);
            let go_left = self.decide_left(idx, fval);
            node = if go_left {
                self.left_child[idx]
            } else {
                self.right_child[idx]
            };
        }
        let leaf_idx = (!node) as usize;
        self.leaf_value.get(leaf_idx).copied().unwrap_or(0.0)
    }

    /// Exact LightGBM C++ node decision logic (`Tree::Decision` in `tree.h`).
    #[inline]
    pub fn decide_left(&self, node: usize, fval: f64) -> bool {
        let dt = self.decision_type[node];
        let is_cat = (dt & 1) != 0;
        if is_cat {
            if fval.is_nan() || fval < 0.0 {
                return false;
            }
            let cat_val = fval as usize;
            let cat_idx = self.threshold[node] as usize;
            if cat_idx + 1 >= self.cat_boundaries.len() {
                return false;
            }
            let start = self.cat_boundaries[cat_idx];
            let end = self.cat_boundaries[cat_idx + 1];
            let word = cat_val / 32;
            let bit = cat_val % 32;
            if start + word < end && start + word < self.cat_threshold.len() {
                ((self.cat_threshold[start + word] >> bit) & 1) != 0
            } else {
                false
            }
        } else {
            let default_left = (dt & 2) != 0;
            let missing_type = (dt >> 2) & 3;
            let is_missing = match missing_type {
                1 => fval.is_nan() || fval.abs() <= 1e-35, // MissingType::Zero
                2 => fval.is_nan(),                        // MissingType::NaN
                _ => false,                                // MissingType::None
            };
            if is_missing {
                default_left
            } else {
                let effective = if fval.is_nan() { 0.0 } else { fval };
                effective <= self.threshold[node]
            }
        }
    }

    pub fn max_depth(&self) -> usize {
        if self.num_leaves <= 1 || self.split_feature.is_empty() {
            return 0;
        }
        fn depth_rec(tree: &Tree, node: i32) -> usize {
            if node < 0 {
                0
            } else {
                let i = node as usize;
                1 + depth_rec(tree, tree.left_child[i]).max(depth_rec(tree, tree.right_child[i]))
            }
        }
        depth_rec(self, 0)
    }
}

#[derive(Debug, Clone)]
pub struct LgbmModel {
    pub version: String,
    pub sha256: String,
    pub objective: Objective,
    pub objective_raw: String,
    pub num_class: usize,
    pub feature_names: Vec<String>,
    /// Map from feature index -> (category_string -> category_code_f64).
    pub categorical_maps: HashMap<usize, HashMap<String, f64>>,
    pub categorical_levels: HashMap<String, Vec<String>>,
    pub trees: Vec<Tree>,
}

impl LgbmModel {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("failed to read LightGBM model {}: {e}", path.display()))?;
        Self::parse(&content)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let sha256: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();

        let mut version = "v4".to_string();
        let mut num_class: usize = 1;
        let mut objective_raw = "regression".to_string();
        let mut feature_names: Vec<String> = Vec::new();
        let mut feature_infos: Vec<String> = Vec::new();
        let mut pandas_cat_raw: Option<String> = None;
        let mut trees: Vec<Tree> = Vec::new();

        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            if let Some(v) = line.strip_prefix("version=") {
                version = v.to_string();
            } else if let Some(v) = line.strip_prefix("num_class=") {
                num_class = v.parse().unwrap_or(1);
            } else if let Some(v) = line.strip_prefix("objective=") {
                objective_raw = v.to_string();
            } else if let Some(v) = line.strip_prefix("feature_names=") {
                feature_names = v.split_whitespace().map(|s| s.to_string()).collect();
            } else if let Some(v) = line.strip_prefix("feature_infos=") {
                feature_infos = v.split_whitespace().map(|s| s.to_string()).collect();
            } else if let Some(v) = line.strip_prefix("pandas_categorical:") {
                pandas_cat_raw = Some(v.trim().to_string());
            } else if line.starts_with("Tree=") {
                let (tree, next_i) = parse_tree_block(&lines, i + 1)?;
                trees.push(tree);
                i = next_i;
                continue;
            }
            i += 1;
        }

        if trees.is_empty() {
            return Err("no `Tree=` sections found in LightGBM model file".to_string());
        }

        let objective = parse_objective(&objective_raw, num_class);

        // Map `pandas_categorical:[[...], ...]` onto categorical feature columns
        // (identified by colon-separated integer codes in `feature_infos`, not `[min:max]`).
        let mut categorical_maps: HashMap<usize, HashMap<String, f64>> = HashMap::new();
        let mut categorical_levels: HashMap<String, Vec<String>> = HashMap::new();
        if let Some(raw_cat) = pandas_cat_raw
            && let Ok(parsed_lists) = serde_json::from_str::<Vec<Vec<serde_json::Value>>>(&raw_cat)
        {
            let cat_feat_indices: Vec<usize> = feature_infos
                .iter()
                .enumerate()
                .filter(|(_, info)| !info.starts_with('[') && info.contains(':') && *info != "none")
                .map(|(idx, _)| idx)
                .collect();

            // Fallback: if `feature_infos` was stripped, locate features used in categorical splits
            let target_indices = if cat_feat_indices.len() == parsed_lists.len() {
                cat_feat_indices
            } else {
                let mut seen = Vec::new();
                for t in &trees {
                    for (node_idx, &dt) in t.decision_type.iter().enumerate() {
                        if (dt & 1) != 0 {
                            let f = t.split_feature[node_idx];
                            if !seen.contains(&f) {
                                seen.push(f);
                            }
                        }
                    }
                }
                seen.sort_unstable();
                seen
            };

            for (feat_idx, levels_val) in target_indices.into_iter().zip(parsed_lists) {
                let mut map = HashMap::new();
                let mut level_names = Vec::new();
                for (code, val) in levels_val.into_iter().enumerate() {
                    let s = match val {
                        serde_json::Value::String(st) => st,
                        other => other.to_string(),
                    };
                    map.insert(s.clone(), code as f64);
                    level_names.push(s);
                }
                if let Some(fname) = feature_names.get(feat_idx) {
                    categorical_levels.insert(fname.clone(), level_names);
                }
                categorical_maps.insert(feat_idx, map);
            }
        }

        Ok(Self {
            version,
            sha256,
            objective,
            objective_raw,
            num_class,
            feature_names,
            categorical_maps,
            categorical_levels,
            trees,
        })
    }

    /// Encode a JSON/string key-value map into an aligned dense `f64` feature vector.
    pub fn encode_record(&self, record: &HashMap<String, String>) -> Vec<f64> {
        let mut x = vec![f64::NAN; self.feature_names.len()];
        for (idx, fname) in self.feature_names.iter().enumerate() {
            let Some(raw) = record.get(fname) else {
                continue;
            };
            let trimmed = raw.trim();
            if trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("nan")
                || trimmed.eq_ignore_ascii_case("null")
                || trimmed.eq_ignore_ascii_case("none")
            {
                continue;
            }
            if let Some(cat_map) = self.categorical_maps.get(&idx) {
                if let Some(&code) = cat_map.get(trimmed) {
                    x[idx] = code;
                } else if let Ok(num) = trimmed.parse::<f64>() {
                    x[idx] = num;
                } else {
                    x[idx] = -1.0; // unseen category
                }
            } else if let Ok(num) = trimmed.parse::<f64>() {
                x[idx] = num;
            } else if trimmed.eq_ignore_ascii_case("true") {
                x[idx] = 1.0;
            } else if trimmed.eq_ignore_ascii_case("false") {
                x[idx] = 0.0;
            }
        }
        x
    }

    /// Compute raw margin score (sum of tree leaf values) for binary/regression.
    #[inline]
    pub fn predict_raw(&self, x: &[f64]) -> f64 {
        self.trees.iter().map(|t| t.predict_row(x)).sum()
    }

    /// Compute transformed prediction (sigmoid for binary, raw for regression, optional Platt).
    #[inline]
    pub fn predict_transformed(
        &self,
        x: &[f64],
        platt: Option<(f64, f64)>,
        anchor: Option<f64>,
    ) -> f64 {
        let raw = self.predict_raw(x);
        match &self.objective {
            Objective::BinarySigmoid => {
                let base_prob = sigmoid(raw);
                if let Some((slope, intercept)) = platt {
                    let c = base_prob.clamp(1e-6, 1.0 - 1e-6);
                    let logit = (c / (1.0 - c)).ln();
                    sigmoid(slope * logit + intercept)
                } else {
                    base_prob
                }
            }
            Objective::Regression | Objective::Quantile | Objective::Other(_) => {
                if let Some(anc) = anchor {
                    let anc_pos = anc.max(1.0);
                    ((anc_pos + 1.0) * raw.clamp(-1.5, 1.5).exp() - 1.0).max(0.0)
                } else {
                    raw
                }
            }
            Objective::Multiclass(_) => self
                .predict_multiclass(x)
                .into_iter()
                .fold(0.0_f64, f64::max),
        }
    }

    /// Compute multiclass softmax probabilities when `num_class > 1`.
    pub fn predict_multiclass(&self, x: &[f64]) -> Vec<f64> {
        let k = self.num_class.max(1);
        let mut logits = vec![0.0; k];
        for (idx, tree) in self.trees.iter().enumerate() {
            logits[idx % k] += tree.predict_row(x);
        }
        let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_e: f64 = exps.iter().sum::<f64>().max(1e-12);
        exps.into_iter().map(|e| e / sum_e).collect()
    }

    /// Global gain-based feature importance across all trees.
    pub fn gain_importance(&self) -> Vec<(String, f64, usize)> {
        let mut gains = vec![0.0_f64; self.feature_names.len()];
        let mut splits = vec![0_usize; self.feature_names.len()];
        for t in &self.trees {
            for (idx, &f) in t.split_feature.iter().enumerate() {
                if f < gains.len() {
                    gains[f] += t.split_gain.get(idx).copied().unwrap_or(0.0);
                    splits[f] += 1;
                }
            }
        }
        let mut out: Vec<(String, f64, usize)> = self
            .feature_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), (gains[i] * 100.0).round() / 100.0, splits[i]))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

#[inline]
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

fn parse_objective(raw: &str, num_class: usize) -> Objective {
    let low = raw.to_lowercase();
    if low.starts_with("binary") {
        Objective::BinarySigmoid
    } else if low.starts_with("multiclass") || num_class > 1 {
        Objective::Multiclass(num_class.max(2))
    } else if low.starts_with("quantile") {
        Objective::Quantile
    } else if low.starts_with("regression") || low.starts_with("huber") || low.starts_with("poisson")
    {
        Objective::Regression
    } else {
        Objective::Other(raw.to_string())
    }
}

fn parse_list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split_whitespace()
        .filter_map(|tok| tok.parse::<T>().ok())
        .collect()
}

fn parse_tree_block(lines: &[&str], start: usize) -> Result<(Tree, usize), String> {
    let mut num_leaves: usize = 1;
    let mut split_feature = Vec::new();
    let mut split_gain = Vec::new();
    let mut threshold = Vec::new();
    let mut decision_type = Vec::new();
    let mut left_child = Vec::new();
    let mut right_child = Vec::new();
    let mut leaf_value = Vec::new();
    let mut leaf_count = Vec::new();
    let mut internal_count = Vec::new();
    let mut cat_boundaries = Vec::new();
    let mut cat_threshold = Vec::new();

    let mut i = start;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("Tree=")
            || line.starts_with("feature_importances:")
            || line.starts_with("parameters:")
        {
            if line.starts_with("Tree=")
                || line.starts_with("feature_importances:")
                || line.starts_with("parameters:")
            {
                break;
            }
            if !leaf_value.is_empty() {
                i += 1;
                break;
            }
            i += 1;
            continue;
        }
        if let Some(v) = line.strip_prefix("num_leaves=") {
            num_leaves = v.parse().unwrap_or(1);
        } else if let Some(v) = line.strip_prefix("split_feature=") {
            split_feature = parse_list(v);
        } else if let Some(v) = line.strip_prefix("split_gain=") {
            split_gain = parse_list(v);
        } else if let Some(v) = line.strip_prefix("threshold=") {
            threshold = parse_list(v);
        } else if let Some(v) = line.strip_prefix("decision_type=") {
            decision_type = parse_list(v);
        } else if let Some(v) = line.strip_prefix("left_child=") {
            left_child = parse_list(v);
        } else if let Some(v) = line.strip_prefix("right_child=") {
            right_child = parse_list(v);
        } else if let Some(v) = line.strip_prefix("leaf_value=") {
            leaf_value = parse_list(v);
        } else if let Some(v) = line.strip_prefix("leaf_count=") {
            leaf_count = parse_list(v);
        } else if let Some(v) = line.strip_prefix("internal_count=") {
            internal_count = parse_list(v);
        } else if let Some(v) = line.strip_prefix("cat_boundaries=") {
            cat_boundaries = parse_list(v);
        } else if let Some(v) = line.strip_prefix("cat_threshold=") {
            cat_threshold = parse_list(v);
        }
        i += 1;
    }

    Ok((
        Tree {
            num_leaves,
            split_feature,
            split_gain,
            threshold,
            decision_type,
            left_child,
            right_child,
            leaf_value,
            leaf_count,
            internal_count,
            cat_boundaries,
            cat_threshold,
        },
        i,
    ))
}
