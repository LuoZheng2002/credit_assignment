use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    direct_answer::generate_raw_answers::LlmModel,
    em::em_types::EmFitResult,
    parallel_process_jsonl::read_json_lines,
    version_tracking::{AssetFile, Base64Hash},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmNodeFit {
    pub node_id: usize,
    pub mean: f64,
    pub log_std: f64,
    /// Display-oriented score for visualization: contribution mean normalized by variance.
    pub mean_div_variance: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmFitPerTree {
    pub tree_question_id: usize,
    pub per_node: Vec<EmNodeFit>,
}

pub fn split_em_fit_result_per_tree(em_fit: &EmFitResult) -> Vec<EmFitPerTree> {
    let mut grouped: BTreeMap<usize, Vec<EmNodeFit>> = BTreeMap::new();
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
            .push(EmNodeFit {
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

pub struct AssetFileEmFitPerTree {
    model: LlmModel,
    dataset: String,
    num_samples: usize,
}

impl AssetFile for AssetFileEmFitPerTree {
    type FileModel = Vec<EmFitPerTree>;
    fn synchronize(&self) -> Base64Hash {
        // it needs to generate em fitting per tree if the file is missing or outdated
        todo!()
    }
    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        read_json_lines(self.file_path()).unwrap()
    }
    fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_em_fit_per_tree_{}.jsonl",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_em_fit_per_tree_{}.tracking.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }
}
