use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::agent::tree_schema::AssetFileTrees;
use crate::{
    agent::tree_schema::CompletedTree,
    direct_answer::generate_raw_answers::LlmModel,
    em::em_dataset_builder::EmDatasetBuilder,
    em::em_fitting::EmFitter,
    em::em_types::EmFitResult,
    em::em_types::{
        EmFitDiagnostics, EmGlobalConfigSnapshot, EmHyperparameters, EmNodeTypePosterior,
    },
    parallel_process_jsonl::{read_json, read_json_lines, write_json, write_jsonl_file},
    version_tracking::{AssetFile, Base64Hash, hash_file},
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmFitMeta {
    #[serde(default)]
    pub ordinary_prior_mean: f64,
    pub global: Vec<EmNodeTypePosterior>,
    pub config: EmGlobalConfigSnapshot,
    pub diagnostics: EmFitDiagnostics,
}

pub fn short_hyperparameter_hash(hyperparameters: &EmHyperparameters) -> String {
    let serialized_hyperparameters =
        serde_json::to_vec(hyperparameters).expect("EmHyperparameters should be serializable");
    let hash = blake3::hash(&serialized_hyperparameters);
    let short_hash = hex::encode(&hash.as_bytes()[..2]);
    assert_eq!(
        short_hash.len(),
        4,
        "Expected a 4-character hexadecimal hyperparameter hash"
    );
    short_hash
}

pub fn run_em_fit(
    completed_trees: &[CompletedTree],
    hyperparameters: EmHyperparameters,
) -> (Vec<EmFitPerTree>, EmFitMeta) {
    assert!(
        !completed_trees.is_empty(),
        "Input trees file must contain at least one CompletedTree entry"
    );

    let mut expected_tree_ids: BTreeSet<usize> = BTreeSet::new();
    for completed_tree in completed_trees {
        assert_eq!(
            completed_tree.id, completed_tree.trajectory.question_id,
            "CompletedTree.id must equal Tree.question_id"
        );
        assert!(
            expected_tree_ids.insert(completed_tree.id),
            "Duplicate CompletedTree.id found in input: {}",
            completed_tree.id
        );
    }

    let dataset = EmDatasetBuilder::new()
        .build_from_tree_iter(completed_trees.iter().map(|tree| &tree.trajectory));
    let fitter = EmFitter::new(hyperparameters);
    let fit_result = fitter.fit(&dataset);
    let per_tree = split_em_fit_result_per_tree(&fit_result);

    assert_eq!(
        per_tree.len(),
        completed_trees.len(),
        "Output EmFitPerTree entries must match input CompletedTree count"
    );
    let actual_tree_ids: BTreeSet<usize> = per_tree
        .iter()
        .map(|entry| entry.tree_question_id)
        .collect();
    assert_eq!(
        actual_tree_ids, expected_tree_ids,
        "Output EmFitPerTree tree id set must match input CompletedTree id set"
    );

    let meta_output = EmFitMeta {
        ordinary_prior_mean: fit_result.config.ordinary_prior_mean,
        global: fit_result.global,
        config: fit_result.config,
        diagnostics: fit_result.diagnostics,
    };
    (per_tree, meta_output)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileEmFitTracking {
    // pub per_tree_file_hash: Base64Hash,
    // pub meta_file_hash: Base64Hash,
    pub trees_hash: Base64Hash,
    pub hyperparameters: EmHyperparameters,
}

pub struct AssetFileEmFit {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
}

impl AssetFileEmFit {
    pub fn hyperparameter_hash(&self) -> String {
        short_hyperparameter_hash(&self.hyperparameters)
    }

    pub fn per_tree_file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_em_fit_per_tree_{}_{}.jsonl",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    pub fn meta_file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_em_fit_meta_{}_{}.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_em_fit_{}_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }
    pub fn store_em_fit_results(&self, per_tree: &[EmFitPerTree], meta: &EmFitMeta) {
        let per_tree_path = self.per_tree_file_path();
        let meta_path = self.meta_file_path();
        write_jsonl_file(&per_tree_path, per_tree).unwrap_or_else(|err| {
            panic!(
                "Failed to write EmFitPerTree output to {}: {}",
                per_tree_path, err
            )
        });
        write_json(&meta_path, meta).unwrap_or_else(|err| {
            panic!(
                "Failed to write EM fit metadata output to {}: {}",
                meta_path, err
            )
        });
    }
}

impl AssetFile for AssetFileEmFit {
    type FileModel = (Vec<EmFitPerTree>, EmFitMeta);

    fn synchronize(&self) -> Base64Hash {
        let asset_file_trees = AssetFileTrees {
            model: self.model.clone(),
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
        };
        let trees_hash = asset_file_trees.synchronize();
        let tracking_content = match read_json::<AssetFileEmFitTracking>(
            &self.version_tracking_path(),
        ) {
            Ok(mut tracking) => {
                if tracking.trees_hash != trees_hash {
                    println!(
                        "[AssetFileEmFit] Detected stale output for model={}, dataset={}, num_samples={}, hyperparameters={:?} due to dependency trees hash mismatch. Regenerating outputs.",
                        self.model.cli_name(),
                        self.dataset,
                        self.num_samples,
                        self.hyperparameters
                    );
                    let completed_trees = asset_file_trees.fetch().load_all().unwrap();
                    let (per_tree, meta) =
                        run_em_fit(&completed_trees, self.hyperparameters.clone());
                    self.store_em_fit_results(&per_tree, &meta);
                    tracking.trees_hash = trees_hash.clone();
                }
                tracking
            }
            Err(_) => {
                println!(
                    "[AssetFileEmFit] No existing tracking file found for model={}, dataset={}, num_samples={}, hyperparameters={:?}. Creating new tracking.",
                    self.model.cli_name(),
                    self.dataset,
                    self.num_samples,
                    self.hyperparameters
                );
                let completed_trees = asset_file_trees.fetch().load_all().unwrap();
                let (per_tree, meta) = run_em_fit(&completed_trees, self.hyperparameters.clone());
                self.store_em_fit_results(&per_tree, &meta);
                AssetFileEmFitTracking {
                    trees_hash,
                    hyperparameters: self.hyperparameters.clone(),
                }
            }
        };
        write_json(self.version_tracking_path(), &tracking_content).unwrap();
        // we only hash one for simplicity
        hash_file(self.per_tree_file_path()).unwrap()
    }

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        let per_tree = read_json_lines(self.per_tree_file_path()).unwrap();
        let meta = read_json(self.meta_file_path()).unwrap();
        (per_tree, meta)
    }
}
