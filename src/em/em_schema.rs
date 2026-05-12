use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    agent::tree_schema::CompletedTree,
    direct_answer::generate_raw_answers::LlmModel,
    em::em_dataset_builder::EmDatasetBuilder,
    em::em_fitting::EmFitter,
    em::em_types::EmFitResult,
    em::em_types::{
        EmFitDiagnostics, EmGlobalConfigSnapshot, EmHyperparameters, EmNodeTypePosterior,
        LogStdClamp,
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
    pub global: Vec<EmNodeTypePosterior>,
    pub config: EmGlobalConfigSnapshot,
    pub diagnostics: EmFitDiagnostics,
}

pub fn get_rollout_trajectory_path(model: LlmModel, dataset: &str, num_samples: usize) -> String {
    format!(
        "results/{}/rollout/{}_trajectory_{}.jsonl",
        model.cli_name(),
        dataset,
        num_samples
    )
}

pub fn get_em_fit_per_tree_path(model: LlmModel, dataset: &str, num_samples: usize) -> String {
    format!(
        "results/{}/rollout/{}_em_fit_per_tree_{}.jsonl",
        model.cli_name(),
        dataset,
        num_samples
    )
}

pub fn get_em_fit_meta_path(model: LlmModel, dataset: &str, num_samples: usize) -> String {
    format!(
        "results/{}/rollout/{}_em_fit_meta_{}.json",
        model.cli_name(),
        dataset,
        num_samples
    )
}

fn default_em_hyperparameters() -> EmHyperparameters {
    EmHyperparameters {
        sigma_ordinary: 1.0,
        sigma_special: 1.0,
        sigma_log_std: 1.0,
        lambda_slack: 1.0,
        eps: 1e-6,
        max_iterations: 100,
        log_std_clamp: LogStdClamp {
            min: -4.0,
            max: 2.0,
        },
    }
}

pub fn run_em_fit_on_completed_trees(
    completed_trees: &[CompletedTree],
    hyperparameters: EmHyperparameters,
) -> (Vec<EmFitPerTree>, EmFitMeta) {
    assert!(
        !completed_trees.is_empty(),
        "Input trees file must contain at least one CompletedTree entry"
    );

    let mut expected_tree_ids: BTreeSet<usize> = BTreeSet::new();
    let mut fit_trees = Vec::with_capacity(completed_trees.len());
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
        fit_trees.push(completed_tree.trajectory.clone());
    }

    let dataset = EmDatasetBuilder::new().build_from_trees(&fit_trees);
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
        global: fit_result.global,
        config: fit_result.config,
        diagnostics: fit_result.diagnostics,
    };
    (per_tree, meta_output)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileEmFitTracking {
    pub per_tree_file_hash: Base64Hash,
    pub meta_file_hash: Base64Hash,
    pub trajectory_hash: Base64Hash,
}

pub struct AssetFileEmFit {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
}

impl AssetFileEmFit {
    pub fn per_tree_file_path(&self) -> String {
        get_em_fit_per_tree_path(self.model, &self.dataset, self.num_samples)
    }

    pub fn meta_file_path(&self) -> String {
        get_em_fit_meta_path(self.model, &self.dataset, self.num_samples)
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/rollout/{}_em_fit_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }
}

fn synchronize_em_fit_outputs(asset_file_em_fit: &AssetFileEmFit) {
    let trajectory_path = get_rollout_trajectory_path(
        asset_file_em_fit.model,
        &asset_file_em_fit.dataset,
        asset_file_em_fit.num_samples,
    );
    let per_tree_path = asset_file_em_fit.per_tree_file_path();
    let meta_path = asset_file_em_fit.meta_file_path();
    let tracking_path = asset_file_em_fit.version_tracking_path();

    let trajectory_hash = hash_file(&trajectory_path).unwrap_or_else(|err| {
        panic!(
            "Failed to hash dependency trajectory file {}: {}",
            trajectory_path, err
        )
    });

    let per_tree_file_exists = Path::new(&per_tree_path).exists();
    let meta_file_exists = Path::new(&meta_path).exists();
    let tracking = read_json::<AssetFileEmFitTracking>(&tracking_path).ok();

    let per_tree_hash_matches_tracking = if per_tree_file_exists {
        let actual_hash = hash_file(&per_tree_path)
            .unwrap_or_else(|err| panic!("Failed to hash file {}: {}", per_tree_path, err));
        match &tracking {
            Some(tracking) => tracking.per_tree_file_hash == actual_hash,
            None => false,
        }
    } else {
        false
    };
    let meta_hash_matches_tracking = if meta_file_exists {
        let actual_hash = hash_file(&meta_path)
            .unwrap_or_else(|err| panic!("Failed to hash file {}: {}", meta_path, err));
        match &tracking {
            Some(tracking) => tracking.meta_file_hash == actual_hash,
            None => false,
        }
    } else {
        false
    };

    let dependency_hash_matches_tracking = match &tracking {
        Some(tracking) => tracking.trajectory_hash == trajectory_hash,
        _ => false,
    };

    let mut stale_reasons: Vec<&str> = Vec::new();
    if !per_tree_file_exists {
        stale_reasons.push("missing per-tree output");
    }
    if !meta_file_exists {
        stale_reasons.push("missing meta output");
    }
    if tracking.is_none() {
        stale_reasons.push("missing tracking file");
    }
    if !per_tree_hash_matches_tracking {
        stale_reasons.push("per-tree hash mismatch");
    }
    if !meta_hash_matches_tracking {
        stale_reasons.push("meta hash mismatch");
    }
    if !dependency_hash_matches_tracking {
        stale_reasons.push("trajectory dependency hash mismatch");
    }

    let stale = !per_tree_file_exists
        || !meta_file_exists
        || !per_tree_hash_matches_tracking
        || !meta_hash_matches_tracking
        || !dependency_hash_matches_tracking;

    if stale {
        println!(
            "[AssetFileEmFit] Regenerating outputs for model={}, dataset={}, num_samples={} ({})",
            asset_file_em_fit.model.cli_name(),
            asset_file_em_fit.dataset,
            asset_file_em_fit.num_samples,
            stale_reasons.join(", ")
        );
        let completed_trees: Vec<CompletedTree> = read_json_lines(&trajectory_path).unwrap_or_else(|err| {
            panic!(
                "Failed to read dependency trajectory file {}: {}",
                trajectory_path,
                err
            )
        });
        let (per_tree, meta) = run_em_fit_on_completed_trees(
            &completed_trees,
            default_em_hyperparameters(),
        );
        write_jsonl_file(&per_tree_path, &per_tree).unwrap_or_else(|err| {
            panic!(
                "Failed to write EmFitPerTree output to {}: {}",
                per_tree_path, err
            )
        });
        write_json(&meta_path, &meta).unwrap_or_else(|err| {
            panic!("Failed to write EM fit metadata output to {}: {}", meta_path, err)
        });

        let per_tree_hash = hash_file(&per_tree_path)
            .unwrap_or_else(|err| panic!("Failed to hash file {}: {}", per_tree_path, err));
        let meta_hash =
            hash_file(&meta_path).unwrap_or_else(|err| panic!("Failed to hash file {}: {}", meta_path, err));

        let tracking_content = AssetFileEmFitTracking {
            per_tree_file_hash: per_tree_hash,
            meta_file_hash: meta_hash,
            trajectory_hash,
        };
        write_json(&tracking_path, &tracking_content).unwrap_or_else(|err| {
            panic!(
                "Failed to write tracking file {}: {}",
                tracking_path, err
            )
        });
    } else {
        println!(
            "[AssetFileEmFit] Up-to-date: model={}, dataset={}, num_samples={}",
            asset_file_em_fit.model.cli_name(),
            asset_file_em_fit.dataset,
            asset_file_em_fit.num_samples
        );
    }
}

impl AssetFile for AssetFileEmFit {
    type FileModel = (Vec<EmFitPerTree>, EmFitMeta);

    fn synchronize(&self) -> Base64Hash {
        synchronize_em_fit_outputs(self);
        hash_file(self.per_tree_file_path()).unwrap()
    }

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        let per_tree = read_json_lines(self.per_tree_file_path()).unwrap();
        let meta = read_json(self.meta_file_path()).unwrap();
        (per_tree, meta)
    }
}
