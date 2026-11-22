//! Exact Lundberg Path-Dependent TreeSHAP (`Algorithm 2`, Lundberg et al. 2018/2020)
//! implemented natively in pure Rust.
//!
//! Uses strictly conserved bottom-up node covers, zero-cover path pruning (prevents
//! `0.0 / 0.0 = NaN` on deep zero-weight subtrees), `average_output` scaling, and
//! per-class tree striding for multiclass models, guaranteeing the efficiency axiom:
//! `expected_value + sum(phi_i) == raw_margin_prediction`.

use crate::lgbm::{LgbmModel, Tree};

#[derive(Debug, Clone, Copy, Default)]
struct PathElem {
    feature_index: i32,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

#[inline]
fn extend_path(
    path: &mut [PathElem],
    unique_depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i32,
) {
    path[unique_depth] = PathElem {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };

    for i in (0..unique_depth).rev() {
        path[i + 1].pweight += one_fraction * path[i].pweight * ((i + 1) as f64)
            / ((unique_depth + 1) as f64);
        path[i].pweight = zero_fraction * path[i].pweight * ((unique_depth - i) as f64)
            / ((unique_depth + 1) as f64);
    }
}

#[inline]
fn unwind_path(path: &mut [PathElem], unique_depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;

    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight =
                next_one_portion * ((unique_depth + 1) as f64) / (((i + 1) as f64) * one_fraction);
            next_one_portion = tmp
                - path[i].pweight * zero_fraction * ((unique_depth - i) as f64)
                    / ((unique_depth + 1) as f64);
        } else if zero_fraction != 0.0 {
            path[i].pweight = (path[i].pweight * ((unique_depth + 1) as f64))
                / (zero_fraction * ((unique_depth - i) as f64));
        } else {
            path[i].pweight = 0.0;
        }
    }

    for i in path_index..unique_depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

#[inline]
fn unwound_path_sum(path: &[PathElem], unique_depth: usize, path_index: usize) -> f64 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let mut total = 0.0;

    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp =
                next_one_portion * ((unique_depth + 1) as f64) / (((i + 1) as f64) * one_fraction);
            total += tmp;
            next_one_portion = path[i].pweight
                - tmp * zero_fraction * ((unique_depth - i) as f64) / ((unique_depth + 1) as f64);
        } else if zero_fraction != 0.0 {
            total += (path[i].pweight / zero_fraction)
                / (((unique_depth - i) as f64) / ((unique_depth + 1) as f64));
        }
    }
    total
}

fn tree_expected_value(tree: &Tree) -> f64 {
    if tree.num_leaves <= 1 || tree.split_feature.is_empty() {
        return tree.leaf_value.first().copied().unwrap_or(0.0);
    }
    let root_cover = tree.node_cover(0);
    if root_cover <= 0.0 {
        return tree.leaf_value.iter().sum::<f64>() / (tree.leaf_value.len().max(1) as f64);
    }
    tree.leaf_value
        .iter()
        .zip(tree.leaf_cover.iter())
        .map(|(&v, &c)| v * c)
        .sum::<f64>()
        / root_cover
}

#[allow(clippy::too_many_arguments)]
fn tree_shap_recurse(
    tree: &Tree,
    x: &[f64],
    node: i32,
    unique_depth: usize,
    parent_path: &[PathElem],
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: i32,
    phi: &mut [f64],
) {
    let mut path = vec![PathElem::default(); unique_depth + 2];
    path[..unique_depth].copy_from_slice(&parent_path[..unique_depth]);
    extend_path(
        &mut path,
        unique_depth,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
    );

    if node < 0 {
        let leaf_idx = (!node) as usize;
        let leaf_val = tree.leaf_value.get(leaf_idx).copied().unwrap_or(0.0);
        for i in 1..=unique_depth {
            let w = unwound_path_sum(&path, unique_depth, i);
            let f_idx = path[i].feature_index as usize;
            if f_idx < phi.len() {
                phi[f_idx] += w * (path[i].one_fraction - path[i].zero_fraction) * leaf_val;
            }
        }
        return;
    }

    let idx = node as usize;
    let feat = tree.split_feature[idx];
    let fval = x.get(feat).copied().unwrap_or(f64::NAN);
    let go_left = tree.decide_left(idx, fval);
    let (hot_child, cold_child) = if go_left {
        (tree.left_child[idx], tree.right_child[idx])
    } else {
        (tree.right_child[idx], tree.left_child[idx])
    };

    let parent_cover = tree.node_cover(node);
    if parent_cover <= 0.0 {
        return;
    }
    let hot_zero_fraction = tree.node_cover(hot_child) / parent_cover;
    let cold_zero_fraction = tree.node_cover(cold_child) / parent_cover;

    let mut incoming_zero_fraction = 1.0;
    let mut incoming_one_fraction = 1.0;

    // If this split feature was already seen higher on the path, unwind it first!
    let mut next_depth = unique_depth;
    if let Some(pos) = (1..=unique_depth).find(|&k| path[k].feature_index == feat as i32) {
        incoming_zero_fraction = path[pos].zero_fraction;
        incoming_one_fraction = path[pos].one_fraction;
        unwind_path(&mut path, unique_depth, pos);
        next_depth = unique_depth - 1;
    }

    let next_hot_zero = hot_zero_fraction * incoming_zero_fraction;
    if next_hot_zero > 0.0 || incoming_one_fraction > 0.0 {
        tree_shap_recurse(
            tree,
            x,
            hot_child,
            next_depth + 1,
            &path,
            next_hot_zero,
            incoming_one_fraction,
            feat as i32,
            phi,
        );
    }

    let next_cold_zero = cold_zero_fraction * incoming_zero_fraction;
    if next_cold_zero > 0.0 {
        tree_shap_recurse(
            tree,
            x,
            cold_child,
            next_depth + 1,
            &path,
            next_cold_zero,
            0.0,
            feat as i32,
            phi,
        );
    }
}

/// Compute exact TreeSHAP values `(expected_value, phi_per_feature)` for one row.
/// For multiclass models (`num_class > 1`), explains the winning predicted class.
pub fn explain_row(model: &LgbmModel, x: &[f64]) -> (f64, Vec<f64>) {
    let mut phi = vec![0.0_f64; model.feature_names.len()];
    let mut expected_value = 0.0_f64;

    let target_class = if model.num_class > 1 {
        let (best_c, _) = model.predict_multiclass_logits(x);
        Some(best_c)
    } else {
        None
    };

    let mut used_trees = 0usize;
    for (tree_idx, tree) in model.trees.iter().enumerate() {
        if let Some(tc) = target_class
            && tree_idx % model.num_class != tc
        {
            continue;
        }
        used_trees += 1;
        expected_value += tree_expected_value(tree);
        if tree.num_leaves <= 1 || tree.split_feature.is_empty() {
            continue;
        }
        let empty: [PathElem; 0] = [];
        tree_shap_recurse(tree, x, 0, 0, &empty, 1.0, 1.0, -1, &mut phi);
    }

    if model.average_output && used_trees > 0 {
        let inv = 1.0 / (used_trees as f64);
        expected_value *= inv;
        for v in &mut phi {
            *v *= inv;
        }
    }

    (expected_value, phi)
}
