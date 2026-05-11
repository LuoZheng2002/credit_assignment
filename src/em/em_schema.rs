use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::em::em_types::EmFitResult;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmNodeFitPerTree {
    pub node_id: usize,
    pub mean: f64,
    pub log_std: f64,
    /// Display-oriented score for visualization: contribution mean normalized by variance.
    pub mean_div_variance: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmFitPerTree {
    pub tree_question_id: usize,
    pub per_node: Vec<EmNodeFitPerTree>,
}

pub fn split_em_fit_result_per_tree(em_fit: &EmFitResult) -> Vec<EmFitPerTree> {
    let mut grouped: BTreeMap<usize, Vec<EmNodeFitPerTree>> = BTreeMap::new();
    let mut seen_per_tree: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();

    for per_node in &em_fit.per_node {
        assert!(
            per_node.mean.is_finite(),
            "per-node contribution mean must be finite"
        );
        assert!(
            per_node.log_std.is_finite(),
            "per-node contribution log_std must be finite"
        );
        let variance = (2.0 * per_node.log_std).exp();
        assert!(
            variance.is_finite() && variance > 0.0,
            "Cannot compute mean/variance with invalid variance (tree={}, node={})",
            per_node.tree_question_id,
            per_node.node_id
        );

        let seen = seen_per_tree.entry(per_node.tree_question_id).or_default();
        assert!(
            seen.insert(per_node.node_id),
            "Duplicate node_id in tree-local EM fit result (tree={}, node={})",
            per_node.tree_question_id,
            per_node.node_id
        );

        grouped
            .entry(per_node.tree_question_id)
            .or_default()
            .push(EmNodeFitPerTree {
                node_id: per_node.node_id,
                mean: per_node.mean,
                log_std: per_node.log_std,
                mean_div_variance: per_node.mean / variance,
            });
    }

    let mut per_tree: Vec<EmFitPerTree> = grouped
        .into_iter()
        .map(|(tree_question_id, mut per_node)| {
            per_node.sort_by_key(|node| node.node_id);
            EmFitPerTree {
                tree_question_id,
                per_node,
            }
        })
        .collect();
    per_tree.sort_by_key(|tree_fit| tree_fit.tree_question_id);
    per_tree
}
