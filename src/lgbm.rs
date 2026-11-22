//! Native pure-Rust LightGBM text booster parser & evaluator.
//!
//! Supports LightGBM v3/v4 text format (`booster.save_model("model.txt")`),
//! with strict structural tree validation (blocks cycles, OOB indices, deep
//! stack overflows, and integer wraparound), numerical & categorical bitset
//! splits, missing-value routing, exact `[categorical_feature: ...]` mapping,
//! binary sigmoid (with `sigmoid:<alpha>`), regression, quantile, exp-link
//! (poisson/gamma/tweedie), random-forest `average_output`, and multiclass.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

const MAX_TREE_DEPTH: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    BinarySigmoid,
    Regression,
    Quantile,
    ExpLink,
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
    #[allow(dead_code)]
    pub leaf_count: Vec<f64>,
    #[allow(dead_code)]
    pub internal_count: Vec<f64>,
    /// Strictly conserved bottom-up node covers (`internal_cover[n] == cover(left) + cover(right)`).
    pub leaf_cover: Vec<f64>,
    pub internal_cover: Vec<f64>,
    pub cat_boundaries: Vec<usize>,
    pub cat_threshold: Vec<u32>,
    pub cached_max_depth: usize,
}

impl Tree {
    /// Evaluate a single validated decision tree on feature vector `x`.
    #[inline]
    pub fn predict_row(&self, x: &[f64]) -> f64 {
        if self.num_leaves <= 1 || self.split_feature.is_empty() {
            return self.leaf_value.first().copied().unwrap_or(0.0);
        }
        let mut node: i32 = 0;
        let max_steps = self.split_feature.len() + 1;
        for _ in 0..max_steps {
            if node < 0 {
                break;
            }
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

    /// Exact LightGBM C++ node decision logic (`Tree::Decision` in `tree.h`),
    /// hardened with checked arithmetic against overflow/wraparound on untrusted models.
    #[inline]
    pub fn decide_left(&self, node: usize, fval: f64) -> bool {
        let dt = self.decision_type[node];
        let is_cat = (dt & 1) != 0;
        if is_cat {
            if !fval.is_finite() || fval < 0.0 || fval.fract() != 0.0 || fval > (u32::MAX as f64) {
                return false;
            }
            let cat_val = fval as usize;
            let raw_thr = self.threshold[node];
            if !raw_thr.is_finite()
                || raw_thr < 0.0
                || raw_thr > ((self.cat_boundaries.len().saturating_sub(1)) as f64)
            {
                return false;
            }
            let cat_idx = raw_thr as usize;
            let Some(next_idx) = cat_idx.checked_add(1) else {
                return false;
            };
            if next_idx >= self.cat_boundaries.len() {
                return false;
            }
            let start = self.cat_boundaries[cat_idx];
            let end = self.cat_boundaries[next_idx];
            let word = cat_val / 32;
            let bit = cat_val % 32;
            if let Some(pos) = start.checked_add(word)
                && pos < end
                && pos < self.cat_threshold.len()
            {
                ((self.cat_threshold[pos] >> bit) & 1) != 0
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
        self.cached_max_depth
    }

    #[inline]
    pub fn node_cover(&self, node: i32) -> f64 {
        if node < 0 {
            let leaf = (!node) as usize;
            self.leaf_cover.get(leaf).copied().unwrap_or(0.0)
        } else {
            self.internal_cover
                .get(node as usize)
                .copied()
                .unwrap_or(0.0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct LgbmModel {
    pub version: String,
    pub sha256: String,
    pub objective: Objective,
    pub objective_raw: String,
    pub sigmoid_alpha: f64,
    pub average_output: bool,
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
        let sha256: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        let mut version = "v4".to_string();
        let mut num_class: usize = 1;
        let mut objective_raw = "regression".to_string();
        let mut average_output = false;
        let mut feature_names: Vec<String> = Vec::new();
        let mut feature_infos: Vec<String> = Vec::new();
        let mut param_cat_features: Option<Vec<usize>> = None;
        let mut pandas_cat_raw: Option<String> = None;
        let mut trees: Vec<Tree> = Vec::new();

        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        let mut in_parameters = false;
        while i < lines.len() {
            let line = lines[i].trim();
            if line == "parameters:" {
                in_parameters = true;
                i += 1;
                continue;
            }
            if line == "end of parameters" {
                in_parameters = false;
                i += 1;
                continue;
            }
            if in_parameters {
                if let Some(inner) = line
                    .strip_prefix("[categorical_feature:")
                    .and_then(|s| s.strip_suffix(']'))
                {
                    let trimmed = inner.trim();
                    if !trimmed.is_empty() {
                        param_cat_features =
                            Some(parse_param_categoricals(trimmed, &feature_names));
                    }
                }
                i += 1;
                continue;
            }

            if let Some(v) = line.strip_prefix("version=") {
                version = v.to_string();
            } else if let Some(v) = line.strip_prefix("num_class=") {
                num_class = v
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("invalid num_class `{v}`"))?
                    .max(1);
            } else if let Some(v) = line.strip_prefix("objective=") {
                objective_raw = v.to_string();
            } else if line == "average_output"
                || line == "average_output=1"
                || line == "average_output=true"
            {
                average_output = true;
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

        // Validate split_feature bounds if feature_names is declared
        if !feature_names.is_empty() {
            for (ti, t) in trees.iter().enumerate() {
                for &f in &t.split_feature {
                    if f >= feature_names.len() {
                        return Err(format!(
                            "Tree={ti} references out-of-bounds split_feature {f} (num_features={})",
                            feature_names.len()
                        ));
                    }
                }
            }
        }

        let (objective, sigmoid_alpha) = parse_objective(&objective_raw, num_class);

        // Map `pandas_categorical:[[...], ...]` onto exact categorical feature columns.
        // Priority 1: `[categorical_feature: ...]` from `parameters:` (exact even when a cat feature is constant/`none`).
        // Priority 2: `feature_infos` entries matching categorical encoding.
        // Priority 3: features used in categorical splits across trees.
        let mut categorical_maps: HashMap<usize, HashMap<String, f64>> = HashMap::new();
        let mut categorical_levels: HashMap<String, Vec<String>> = HashMap::new();
        if let Some(raw_cat) = pandas_cat_raw
            && let Ok(parsed_lists) = serde_json::from_str::<Vec<Vec<serde_json::Value>>>(&raw_cat)
            && !parsed_lists.is_empty()
        {
            let target_indices: Vec<usize> = if let Some(p_cats) = param_cat_features
                && p_cats.len() == parsed_lists.len()
            {
                p_cats
            } else {
                let info_cats: Vec<usize> = feature_infos
                    .iter()
                    .enumerate()
                    .filter(|(_, info)| {
                        !info.starts_with('[') && info.contains(':') && *info != "none"
                    })
                    .map(|(idx, _)| idx)
                    .collect();
                if info_cats.len() == parsed_lists.len() {
                    info_cats
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
                    if seen.len() == parsed_lists.len() {
                        seen
                    } else {
                        // Last resort: preserve positional correspondence using `none` + cat infos
                        let info_with_none: Vec<usize> = feature_infos
                            .iter()
                            .enumerate()
                            .filter(|(_, info)| {
                                !info.starts_with('[') && (info.contains(':') || *info == "none")
                            })
                            .map(|(idx, _)| idx)
                            .collect();
                        if info_with_none.len() == parsed_lists.len() {
                            info_with_none
                        } else {
                            seen
                        }
                    }
                }
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
            sigmoid_alpha,
            average_output,
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
                } else if let Ok(num) = trimmed.parse::<f64>()
                    && num.is_finite()
                {
                    x[idx] = num;
                } else {
                    x[idx] = -1.0; // unseen category
                }
            } else if let Ok(num) = trimmed.parse::<f64>() {
                if num.is_finite() {
                    x[idx] = num;
                }
            } else if trimmed.eq_ignore_ascii_case("true") {
                x[idx] = 1.0;
            } else if trimmed.eq_ignore_ascii_case("false") {
                x[idx] = 0.0;
            }
        }
        x
    }

    /// Compute per-class raw logits for multiclass (`num_class > 1`) and winning class index.
    pub fn predict_multiclass_logits(&self, x: &[f64]) -> (usize, Vec<f64>) {
        let k = self.num_class.max(1);
        let mut logits = vec![0.0_f64; k];
        let mut counts = vec![0usize; k];
        for (idx, tree) in self.trees.iter().enumerate() {
            let c = idx % k;
            logits[c] += tree.predict_row(x);
            counts[c] += 1;
        }
        if self.average_output {
            for c in 0..k {
                if counts[c] > 0 {
                    logits[c] /= counts[c] as f64;
                }
            }
        }
        let best_class = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best_class, logits)
    }

    /// Compute raw margin score (sum/average of tree leaf values).
    /// For multiclass (`num_class > 1`), returns the winning class's raw logit.
    #[inline]
    pub fn predict_raw(&self, x: &[f64]) -> f64 {
        if self.num_class > 1 {
            let (best_c, logits) = self.predict_multiclass_logits(x);
            return logits[best_c];
        }
        let sum: f64 = self.trees.iter().map(|t| t.predict_row(x)).sum();
        if self.average_output && !self.trees.is_empty() {
            sum / (self.trees.len() as f64)
        } else {
            sum
        }
    }

    /// Compute transformed prediction (sigmoid/Platt for binary, exp for poisson/gamma/tweedie,
    /// anchored log-ratio for regression when `--anchor-col` is given, softmax max for multiclass).
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
                let z = self.sigmoid_alpha * raw;
                if let Some((slope, intercept)) = platt {
                    sigmoid(slope * z + intercept)
                } else {
                    sigmoid(z)
                }
            }
            Objective::ExpLink => raw.clamp(-700.0, 700.0).exp(),
            Objective::Regression | Objective::Quantile | Objective::Other(_) => {
                if let Some(anc) = anchor.filter(|v| v.is_finite()) {
                    let anc_nonneg = anc.max(0.0);
                    ((anc_nonneg + 1.0) * raw.clamp(-20.0, 20.0).exp() - 1.0).max(0.0)
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
        let (_, logits) = self.predict_multiclass_logits(x);
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

/// Overflow-safe logistic sigmoid (no artificial clamping of extreme log-odds).
#[inline]
pub fn sigmoid(z: f64) -> f64 {
    if !z.is_finite() {
        return if z.is_sign_positive() { 1.0 } else { 0.0 };
    }
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn parse_param_categoricals(spec: &str, feature_names: &[String]) -> Vec<usize> {
    let mut out = Vec::new();
    let trimmed = spec.trim();
    if let Some(names_part) = trimmed.strip_prefix("name:") {
        for raw_name in names_part.split(',') {
            let n = raw_name.trim();
            if let Some(idx) = feature_names.iter().position(|f| f == n) {
                out.push(idx);
            }
        }
    } else {
        for tok in trimmed.split(',') {
            if let Ok(idx) = tok.trim().parse::<usize>() {
                out.push(idx);
            }
        }
    }
    out
}

fn parse_objective(raw: &str, num_class: usize) -> (Objective, f64) {
    let low = raw.to_lowercase();
    let mut alpha = 1.0_f64;
    for tok in low.split_whitespace() {
        if let Some(v) = tok.strip_prefix("sigmoid:")
            && let Ok(a) = v.parse::<f64>()
            && a.is_finite()
            && a > 0.0
        {
            alpha = a;
        }
    }
    let obj = if low.starts_with("binary") {
        Objective::BinarySigmoid
    } else if low.starts_with("multiclass") || num_class > 1 {
        Objective::Multiclass(num_class.max(2))
    } else if low.starts_with("quantile") {
        Objective::Quantile
    } else if low.starts_with("poisson") || low.starts_with("gamma") || low.starts_with("tweedie") {
        Objective::ExpLink
    } else if low.starts_with("regression")
        || low.starts_with("huber")
        || low.starts_with("l1")
        || low.starts_with("l2")
    {
        Objective::Regression
    } else {
        Objective::Other(raw.to_string())
    };
    (obj, alpha)
}

fn parse_list<T: std::str::FromStr>(field: &str, s: &str) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        let val = tok
            .parse::<T>()
            .map_err(|_| format!("invalid token `{tok}` in LightGBM tree field `{field}`"))?;
        out.push(val);
    }
    Ok(out)
}

fn parse_tree_block(lines: &[&str], start: usize) -> Result<(Tree, usize), String> {
    let mut num_leaves: usize = 1;
    let mut split_feature: Vec<usize> = Vec::new();
    let mut split_gain: Vec<f64> = Vec::new();
    let mut threshold: Vec<f64> = Vec::new();
    let mut decision_type: Vec<u8> = Vec::new();
    let mut left_child: Vec<i32> = Vec::new();
    let mut right_child: Vec<i32> = Vec::new();
    let mut leaf_value: Vec<f64> = Vec::new();
    let mut leaf_count: Vec<f64> = Vec::new();
    let mut internal_count: Vec<f64> = Vec::new();
    let mut cat_boundaries: Vec<usize> = Vec::new();
    let mut cat_threshold: Vec<u32> = Vec::new();

    let mut i = start;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("Tree=")
            || line.starts_with("feature_importances:")
            || line.starts_with("parameters:")
            || line.starts_with("end of trees")
        {
            if line.starts_with("Tree=")
                || line.starts_with("feature_importances:")
                || line.starts_with("parameters:")
                || line.starts_with("end of trees")
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
            num_leaves = v
                .trim()
                .parse()
                .map_err(|_| format!("invalid num_leaves `{v}`"))?;
        } else if let Some(v) = line.strip_prefix("is_linear=") {
            if v.trim() == "1" || v.trim().eq_ignore_ascii_case("true") {
                return Err("LightGBM linear trees (is_linear=1) are not supported".to_string());
            }
        } else if let Some(v) = line.strip_prefix("split_feature=") {
            split_feature = parse_list("split_feature", v)?;
        } else if let Some(v) = line.strip_prefix("split_gain=") {
            split_gain = parse_list("split_gain", v)?;
        } else if let Some(v) = line.strip_prefix("threshold=") {
            threshold = parse_list("threshold", v)?;
        } else if let Some(v) = line.strip_prefix("decision_type=") {
            decision_type = parse_list("decision_type", v)?;
        } else if let Some(v) = line.strip_prefix("left_child=") {
            left_child = parse_list("left_child", v)?;
        } else if let Some(v) = line.strip_prefix("right_child=") {
            right_child = parse_list("right_child", v)?;
        } else if let Some(v) = line.strip_prefix("leaf_value=") {
            leaf_value = parse_list("leaf_value", v)?;
        } else if let Some(v) = line.strip_prefix("leaf_count=") {
            leaf_count = parse_list("leaf_count", v)?;
        } else if let Some(v) = line.strip_prefix("internal_count=") {
            internal_count = parse_list("internal_count", v)?;
        } else if let Some(v) = line.strip_prefix("cat_boundaries=") {
            cat_boundaries = parse_list("cat_boundaries", v)?;
        } else if let Some(v) = line.strip_prefix("cat_threshold=") {
            cat_threshold = parse_list("cat_threshold", v)?;
        }
        i += 1;
    }

    // Strict structural validation on untrusted LightGBM tree blocks (P2-CRIT-03)
    if num_leaves == 0 {
        return Err("invalid LightGBM tree: num_leaves must be >= 1".to_string());
    }
    if leaf_value.len() != num_leaves {
        return Err(format!(
            "invalid LightGBM tree: leaf_value.len()={} != num_leaves={num_leaves}",
            leaf_value.len()
        ));
    }
    if leaf_value.iter().any(|v| !v.is_finite()) {
        return Err("invalid LightGBM tree: non-finite leaf_value".to_string());
    }

    let n_int = num_leaves - 1;
    if n_int > 0
        && (split_feature.len() != n_int
            || threshold.len() != n_int
            || decision_type.len() != n_int
            || left_child.len() != n_int
            || right_child.len() != n_int)
    {
        return Err(format!(
            "invalid LightGBM tree: internal array length mismatch (expected {n_int})"
        ));
    }

    // Validate cat_boundaries monotonicity and bounds
    for win in cat_boundaries.windows(2) {
        if win[0] > win[1] {
            return Err("invalid LightGBM tree: cat_boundaries must be non-decreasing".to_string());
        }
    }
    if let Some(&last_b) = cat_boundaries.last()
        && last_b > cat_threshold.len()
    {
        return Err(format!(
            "invalid LightGBM tree: cat_boundaries end ({last_b}) exceeds cat_threshold.len() ({})",
            cat_threshold.len()
        ));
    }

    // Iterative DFS validation: verify strict acyclic tree, child bounds, and max depth,
    // while recording post-order for exact bottom-up cover conservation (P2-HIGH-03).
    let mut cached_max_depth = 0usize;
    let mut post_order: Vec<usize> = Vec::with_capacity(n_int);
    if n_int > 0 {
        let mut visited_int = vec![false; n_int];
        let mut visited_leaf = vec![false; num_leaves];
        // (node_id, depth, expanded)
        let mut stack: Vec<(i32, usize, bool)> = vec![(0, 0, false)];
        while let Some((node, depth, expanded)) = stack.pop() {
            if depth > MAX_TREE_DEPTH {
                return Err(format!(
                    "invalid LightGBM tree: depth exceeds safety limit ({MAX_TREE_DEPTH})"
                ));
            }
            if node < 0 {
                let leaf_idx = (!node) as usize;
                if leaf_idx >= num_leaves {
                    return Err(format!(
                        "invalid LightGBM tree: leaf child index {leaf_idx} out of bounds (num_leaves={num_leaves})"
                    ));
                }
                if visited_leaf[leaf_idx] {
                    return Err(format!(
                        "invalid LightGBM tree: leaf {leaf_idx} is reachable via multiple paths"
                    ));
                }
                visited_leaf[leaf_idx] = true;
                cached_max_depth = cached_max_depth.max(depth);
            } else {
                let idx = node as usize;
                if idx >= n_int {
                    return Err(format!(
                        "invalid LightGBM tree: internal child index {idx} out of bounds (n_int={n_int})"
                    ));
                }
                if expanded {
                    post_order.push(idx);
                } else {
                    if visited_int[idx] {
                        return Err(format!(
                            "invalid LightGBM tree: cycle or shared internal node detected at node {idx}"
                        ));
                    }
                    visited_int[idx] = true;
                    stack.push((node, depth, true));
                    stack.push((right_child[idx], depth + 1, false));
                    stack.push((left_child[idx], depth + 1, false));
                }
            }
        }
        if post_order.len() != n_int || !visited_leaf.iter().all(|&v| v) {
            return Err("invalid LightGBM tree: disconnected internal or leaf nodes".to_string());
        }
    }

    // Precompute strictly conserved bottom-up node covers (`cover[parent] == cover[left] + cover[right]`)
    let mut leaf_cover: Vec<f64> = (0..num_leaves)
        .map(|l| {
            leaf_count
                .get(l)
                .copied()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(0.0)
        })
        .collect();
    let total_leaf_cover: f64 = leaf_cover.iter().sum();
    if total_leaf_cover <= 0.0 {
        leaf_cover.fill(1.0);
    }
    let mut internal_cover = vec![0.0_f64; n_int];
    for idx in post_order {
        let lc = left_child[idx];
        let rc = right_child[idx];
        let lcov = if lc < 0 {
            leaf_cover[(!lc) as usize]
        } else {
            internal_cover[lc as usize]
        };
        let rcov = if rc < 0 {
            leaf_cover[(!rc) as usize]
        } else {
            internal_cover[rc as usize]
        };
        internal_cover[idx] = lcov + rcov;
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
            leaf_cover,
            internal_cover,
            cat_boundaries,
            cat_threshold,
            cached_max_depth,
        },
        i,
    ))
}
