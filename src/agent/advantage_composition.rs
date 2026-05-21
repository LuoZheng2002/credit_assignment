use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    agent::action_log_schema::AssetFileActionLogs,
    agent::trajectory_action_types::StepQuality,
    agent::tree_schema::CompletedTree,
    asset_file::{AssetFile, Base64Hash, hash_file},
    em::{
        em_schema::{AssetFileEmFit, EmFitPerTree, short_hyperparameter_hash},
        em_types::EmHyperparameters,
    },
    json_line_util::{read_json, write_json},
    llm_model::LlmModelName,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdvantageCompositionPerTree {
    pub question_id: usize,
    pub per_node: Vec<AdvantageCompositionPerNode>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdvantageCompositionPerNode {
    pub node_id: usize,
    pub contribution_mean: f64, // when displayed, normalize only the scaling factor, and set pure red / green color beyond 95% ci
    pub contribution_log_std: f64, // when displayed, normalize both the mean and std, and set pure red / green color beyond 95% ci
    pub contribution_mean_div_var: f64, // when displayed, normalize only the scaling factor, and set pure red / green color beyond 95% ci
    pub contribution_mean_div_var_normalized: f64, // this already normalizes both the mean and std within the tree, then multiplied by the tree-level win_loss_ratio_factor. For display, it should be multiplied by a weight factor, and then set pure red / green color beyond 95% ci of N(0, 1)
    pub step_quality_tool_advantage_normalized: f64, // normalized across all trees to N(0, 1). For display, multiplied by a weight factor, and then set pure red / green color beyond 95% ci of N(0, 1)
    pub step_quality_complete_advantage_normalized: f64, // same as above
    pub step_quality_focused_advantage_normalized: f64, // same as above
    pub trajectory_advantage_normalized: f64,        // same as above
}

pub struct AssetFileAdvantageComposition {
    pub model: LlmModelName,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileAdvantageCompositionTracking {
    pub trees_hash: Base64Hash,
    pub em_fit_hash: Base64Hash,
}

impl AssetFileAdvantageComposition {
    pub fn hyperparameter_hash(&self) -> String {
        short_hyperparameter_hash(&self.hyperparameters)
    }

    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_advantage_composition_{}_{}.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_advantage_composition_{}_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    fn trajectory_length_factor(average_trajectory_length: f64) -> f64 {
        assert!(
            average_trajectory_length.is_finite() && average_trajectory_length >= 1.0,
            "average_trajectory_length must be finite and >= 1.0"
        );
        if average_trajectory_length <= 6.0 {
            (average_trajectory_length - 1.0) / 5.0
        } else {
            let decay = std::f64::consts::LN_2 / 6.0;
            (-(average_trajectory_length - 6.0) * decay).exp()
        }
    }

    fn win_loss_ratio_factor(tree_accuracy: f64) -> f64 {
        assert!(
            tree_accuracy.is_finite() && (0.0..=1.0).contains(&tree_accuracy),
            "tree_accuracy must be finite and within [0, 1]"
        );
        1.0 - 2.0 * (tree_accuracy - 0.5).abs()
    }

    fn average_leaf_path_length(tree: &CompletedTree) -> f64 {
        let trajectory = &tree.trajectory;
        assert!(
            !trajectory.leaf_node_ids.is_empty(),
            "Tree {} must have at least one leaf trajectory",
            tree.id
        );
        let mut total_length = 0usize;
        for &leaf_node_id in &trajectory.leaf_node_ids {
            let mut path_length = 0usize;
            let mut seen: BTreeSet<usize> = BTreeSet::new();
            let mut cursor = Some(leaf_node_id);
            while let Some(node_id) = cursor {
                assert!(
                    node_id < trajectory.nodes.len(),
                    "Tree {} has out-of-bounds node id {} on leaf path",
                    tree.id,
                    node_id
                );
                assert!(
                    seen.insert(node_id),
                    "Tree {} leaf path for node {} contains a cycle",
                    tree.id,
                    leaf_node_id
                );
                path_length += 1;
                cursor = trajectory.nodes[node_id].parent_id;
            }
            assert!(path_length > 0, "Leaf path length must be positive");
            total_length += path_length;
        }
        total_length as f64 / trajectory.leaf_node_ids.len() as f64
    }

    fn normalize_to_standard(values: &[f64], name: &str) -> Vec<f64> {
        assert!(
            !values.is_empty(),
            "Cannot normalize empty value list for {}",
            name
        );
        for &value in values {
            assert!(
                value.is_finite(),
                "All values for {} normalization must be finite",
                name
            );
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f64>()
            / values.len() as f64;
        assert!(
            variance.is_finite() && variance >= 0.0,
            "Cannot normalize {} because variance is negative or non-finite",
            name
        );
        if variance == 0.0 {
            return vec![0.0; values.len()];
        }
        let std = variance.sqrt();
        values.iter().map(|value| (*value - mean) / std).collect()
    }

    pub fn compose_advantage(
        trees: &[CompletedTree],
        em_fit_per_tree: &[EmFitPerTree],
    ) -> Vec<AdvantageCompositionPerTree> {
        let mut em_by_tree_id: BTreeMap<usize, &EmFitPerTree> = BTreeMap::new();
        for tree_fit in em_fit_per_tree {
            assert!(
                em_by_tree_id
                    .insert(tree_fit.tree_question_id, tree_fit)
                    .is_none(),
                "Duplicate tree id in EM fit input: {}",
                tree_fit.tree_question_id
            );
        }
        assert!(
            !em_by_tree_id.is_empty(),
            "Advantage composition requires at least one tree"
        );

        let em_tree_ids: BTreeSet<usize> = em_by_tree_id.keys().copied().collect();

        let mut tree_ids: BTreeSet<usize> = BTreeSet::new();
        let mut tree_length_factor_raw_by_tree_id: BTreeMap<usize, f64> = BTreeMap::new();
        let mut win_loss_ratio_factor_by_tree_id: BTreeMap<usize, f64> = BTreeMap::new();

        for tree in trees {
            assert_eq!(
                tree.id, tree.trajectory.question_id,
                "CompletedTree.id must equal Tree.question_id"
            );
            assert!(
                tree_ids.insert(tree.id),
                "Duplicate tree id in trees input: {}",
                tree.id
            );
            assert!(
                em_by_tree_id.contains_key(&tree.id),
                "Tree id {} from input trees missing in EM fit input",
                tree.id
            );

            let average_trajectory_length = Self::average_leaf_path_length(&tree);
            let factor = Self::trajectory_length_factor(average_trajectory_length);
            assert!(
                factor.is_finite(),
                "Trajectory length factor must be finite for tree {}",
                tree.id
            );
            tree_length_factor_raw_by_tree_id.insert(tree.id, factor);

            let correctness_ratio = &tree.trajectory.correctness_ratio;
            assert!(
                correctness_ratio.denominator > 0,
                "Tree {} correctness_ratio denominator must be > 0",
                tree.id
            );
            assert!(
                correctness_ratio.numerator <= correctness_ratio.denominator,
                "Tree {} correctness_ratio numerator must be <= denominator",
                tree.id
            );
            let tree_accuracy =
                correctness_ratio.numerator as f64 / correctness_ratio.denominator as f64;
            let win_loss_ratio_factor = Self::win_loss_ratio_factor(tree_accuracy);
            assert!(
                win_loss_ratio_factor.is_finite(),
                "win_loss_ratio_factor must be finite for tree {}",
                tree.id
            );
            win_loss_ratio_factor_by_tree_id.insert(tree.id, win_loss_ratio_factor);
        }
        assert!(
            !tree_ids.is_empty(),
            "Advantage composition requires at least one tree"
        );

        assert_eq!(
            tree_ids, em_tree_ids,
            "Tree id set must match EM fit tree id set"
        );

        let tree_length_factors_raw: Vec<f64> = tree_length_factor_raw_by_tree_id
            .values()
            .copied()
            .collect();
        let tree_length_factors_normalized = Self::normalize_to_standard(
            &tree_length_factors_raw,
            "trajectory length factors across trees",
        );
        let mut tree_length_factor_normalized_by_tree_id: BTreeMap<usize, f64> = BTreeMap::new();
        for (index, tree_id) in tree_length_factor_raw_by_tree_id
            .keys()
            .copied()
            .enumerate()
        {
            tree_length_factor_normalized_by_tree_id
                .insert(tree_id, tree_length_factors_normalized[index]);
        }

        let mut output: Vec<AdvantageCompositionPerTree> = Vec::with_capacity(tree_ids.len());
        let mut tool_values_raw: Vec<f64> = Vec::new();
        let mut complete_values_raw: Vec<f64> = Vec::new();
        let mut focused_values_raw: Vec<f64> = Vec::new();
        let mut node_positions: Vec<(usize, usize)> = Vec::new();

        for tree in trees {
            let tree_id = tree.id;
            let tree_fit = em_by_tree_id
                .get(&tree_id)
                .expect("Tree id existence already validated");
            let trajectory = &tree.trajectory;

            let mut step_quality_by_node_id: BTreeMap<usize, (f64, f64, f64)> = BTreeMap::new();
            for node in &trajectory.nodes {
                assert_eq!(
                    trajectory.nodes[node.node_id].node_id, node.node_id,
                    "Tree {} node index must equal node_id",
                    tree_id
                );
                let step_quality = node
                    .step
                    .get_step_quality()
                    .expect("Each node must have step quality for advantage composition");
                let (tool, complete, focused) = match step_quality {
                    StepQuality::ProperlyEnded {
                        tool,
                        complete,
                        focused,
                    } => (
                        if tool { 1.0 } else { 0.0 },
                        if complete { 1.0 } else { 0.0 },
                        if focused { 1.0 } else { 0.0 },
                    ),
                    StepQuality::FailedAndAborted => (0.0, 0.0, 0.0),
                };
                assert!(
                    step_quality_by_node_id
                        .insert(node.node_id, (tool, complete, focused))
                        .is_none(),
                    "Duplicate node id {} in tree {}",
                    node.node_id,
                    tree_id
                );
            }

            let contribution_mean_div_var_values: Vec<f64> = tree_fit
                .per_node
                .iter()
                .map(|node_fit| node_fit.mean_div_variance)
                .collect();
            let contribution_normalized_values = Self::normalize_to_standard(
                &contribution_mean_div_var_values,
                &format!("contribution mean_div_variance in tree {}", tree_id),
            );
            let mut contribution_normalized_by_node_id: BTreeMap<usize, f64> = BTreeMap::new();
            for (index, node_fit) in tree_fit.per_node.iter().enumerate() {
                assert!(
                    contribution_normalized_by_node_id
                        .insert(node_fit.node_id, contribution_normalized_values[index])
                        .is_none(),
                    "Duplicate node id {} in EM fit tree {}",
                    node_fit.node_id,
                    tree_id
                );
            }

            let em_node_ids: BTreeSet<usize> =
                tree_fit.per_node.iter().map(|node| node.node_id).collect();
            let tree_node_ids: BTreeSet<usize> =
                trajectory.nodes.iter().map(|node| node.node_id).collect();
            assert_eq!(
                em_node_ids, tree_node_ids,
                "EM node id set must match trajectory node id set in tree {}",
                tree_id
            );

            let trajectory_advantage_normalized = *tree_length_factor_normalized_by_tree_id
                .get(&tree_id)
                .expect("Tree id existence already validated for trajectory factors");
            let win_loss_ratio_factor = *win_loss_ratio_factor_by_tree_id
                .get(&tree_id)
                .expect("Tree id existence already validated for win/loss ratio factors");

            let mut per_node = Vec::with_capacity(tree_fit.per_node.len());
            for node_fit in &tree_fit.per_node {
                let (tool, complete, focused) = *step_quality_by_node_id
                    .get(&node_fit.node_id)
                    .expect("Step quality map should contain all nodes present in EM fit");
                let contribution_mean_div_var_normalized = *contribution_normalized_by_node_id
                    .get(&node_fit.node_id)
                    .expect("Contribution normalization map should contain all EM nodes");
                let contribution_mean_div_var_normalized =
                    contribution_mean_div_var_normalized * win_loss_ratio_factor;
                per_node.push(AdvantageCompositionPerNode {
                    node_id: node_fit.node_id,
                    contribution_mean: node_fit.mean,
                    contribution_log_std: node_fit.log_std,
                    contribution_mean_div_var: node_fit.mean_div_variance,
                    contribution_mean_div_var_normalized,
                    step_quality_tool_advantage_normalized: 0.0,
                    step_quality_complete_advantage_normalized: 0.0,
                    step_quality_focused_advantage_normalized: 0.0,
                    trajectory_advantage_normalized,
                });
                tool_values_raw.push(tool);
                complete_values_raw.push(complete);
                focused_values_raw.push(focused);
                node_positions.push((output.len(), per_node.len() - 1));
            }

            output.push(AdvantageCompositionPerTree {
                question_id: tree_id,
                per_node,
            });
        }

        assert_eq!(tool_values_raw.len(), node_positions.len());
        assert_eq!(complete_values_raw.len(), node_positions.len());
        assert_eq!(focused_values_raw.len(), node_positions.len());

        let tool_values_normalized =
            Self::normalize_to_standard(&tool_values_raw, "tool step quality across nodes");
        let complete_values_normalized =
            Self::normalize_to_standard(&complete_values_raw, "complete step quality across nodes");
        let focused_values_normalized =
            Self::normalize_to_standard(&focused_values_raw, "focused step quality across nodes");

        for (global_index, (tree_index, node_index)) in node_positions.into_iter().enumerate() {
            let node = &mut output[tree_index].per_node[node_index];
            node.step_quality_tool_advantage_normalized = tool_values_normalized[global_index];
            node.step_quality_complete_advantage_normalized =
                complete_values_normalized[global_index];
            node.step_quality_focused_advantage_normalized =
                focused_values_normalized[global_index];
        }

        output
    }
}

impl AssetFile for AssetFileAdvantageComposition {
    type FileModel = Vec<AdvantageCompositionPerTree>;

    fn synchronize(&self) -> Base64Hash {
        let action_logs = AssetFileActionLogs {
            model: self.model.clone(),
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
        };
        let trees_hash = hash_file(action_logs.file_path()).unwrap();

        let asset_file_em_fit = AssetFileEmFit {
            model: self.model.clone(),
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
            hyperparameters: self.hyperparameters.clone(),
        };
        let em_fit_hash = asset_file_em_fit.synchronize();

        let tracking_content = match read_json::<AssetFileAdvantageCompositionTracking>(
            self.version_tracking_path(),
        ) {
            Ok(mut tracking) => {
                if tracking.trees_hash != trees_hash || tracking.em_fit_hash != em_fit_hash {
                    println!(
                        "[AssetFileAdvantageComposition] Detected stale output for model={}, dataset={}, num_samples={}, hyperparameters={:?}. Regenerating outputs.",
                        self.model.cli_name(),
                        self.dataset,
                        self.num_samples,
                        self.hyperparameters
                    );
                    let trees = action_logs.load_completed_trees_sync();
                    let (em_fit_per_tree, _meta) = asset_file_em_fit.fetch();
                    let advantage = Self::compose_advantage(&trees, &em_fit_per_tree);
                    write_json(self.file_path(), &advantage).unwrap();
                    tracking.trees_hash = trees_hash.clone();
                    tracking.em_fit_hash = em_fit_hash.clone();
                }
                tracking
            }
            Err(_) => {
                println!(
                    "[AssetFileAdvantageComposition] No existing tracking file found for model={}, dataset={}, num_samples={}, hyperparameters={:?}. Creating new tracking.",
                    self.model.cli_name(),
                    self.dataset,
                    self.num_samples,
                    self.hyperparameters
                );
                let trees = action_logs.load_completed_trees_sync();
                let (em_fit_per_tree, _meta) = asset_file_em_fit.fetch();
                let advantage = Self::compose_advantage(&trees, &em_fit_per_tree);
                write_json(self.file_path(), &advantage).unwrap();
                AssetFileAdvantageCompositionTracking {
                    trees_hash,
                    em_fit_hash,
                }
            }
        };
        write_json(self.version_tracking_path(), &tracking_content).unwrap();
        hash_file(self.file_path()).unwrap()
    }

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        read_json(self.file_path()).unwrap()
    }
}
