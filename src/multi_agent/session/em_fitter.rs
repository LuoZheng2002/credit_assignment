use serde::{Deserialize, Serialize};

use super::types::VerifierAndModeSummary;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeafLabel {
    Correct,
    Incorrect,
}

impl LeafLabel {
    pub fn as_sign(self) -> f64 {
        match self {
            LeafLabel::Correct => 1.0,
            LeafLabel::Incorrect => -1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStdClamp {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmHyperparameters {
    pub sigma_mean: f64,
    pub sigma_mode: f64,
    pub sigma_log_std: f64,
    pub mu_log_std_mode: f64,
    pub lambda_slack: f64,
    pub eps: f64,
    pub max_iterations: usize,
    pub log_std_clamp: LogStdClamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmNodeBinding {
    pub global_node_id: usize,
    pub tree_question_id: usize,
    pub node_id: usize,
    pub mode: VerifierAndModeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmLeafBinding {
    pub global_leaf_id: usize,
    pub tree_question_id: usize,
    pub leaf_node_id: usize,
    pub label: LeafLabel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsePathTerm {
    pub global_node_id: usize,
    pub x_li: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmLeafPath {
    pub global_leaf_id: usize,
    pub terms: Vec<SparsePathTerm>,
}

/// Interaction boundary between tree logs and global EM fitting.
///
/// This struct is the tree-facing extracted representation:
/// - one flat global node index space over all selected trees,
/// - one flat global judged-leaf index space,
/// - sparse path encoding x_{l,i} for constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmFitDataset {
    pub node_bindings: Vec<EmNodeBinding>,
    pub leaf_bindings: Vec<EmLeafBinding>,
    pub leaf_paths: Vec<EmLeafPath>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmNodePosterior {
    pub global_node_id: usize,
    pub tree_question_id: usize,
    pub node_id: usize,
    pub mean: f64,
    pub log_std: f64,
    pub mode: VerifierAndModeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmModePosterior {
    pub mode: VerifierAndModeSummary,
    pub mu_k: f64,
    pub nu_k: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmConstraintDiagnostics {
    pub sum_xi: f64,
    pub num_positive_xi: usize,
    /// Largest slack violators sorted descending by slack.
    pub largest_violators: Vec<EmLeafSlack>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmLeafSlack {
    pub global_leaf_id: usize,
    pub tree_question_id: usize,
    pub leaf_node_id: usize,
    pub slack: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmFitDiagnostics {
    pub objective_trace: Vec<f64>,
    pub converged_flag: bool,
    pub final_train_sign_accuracy: f64,
    /// Validation is deferred for now; kept for schema compatibility.
    pub final_val_sign_accuracy: Option<f64>,
    pub mean_slack_train: f64,
    /// Validation is deferred for now; kept for schema compatibility.
    pub mean_slack_val: Option<f64>,
    pub constraints: EmConstraintDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmGlobalConfigSnapshot {
    pub hyperparameters: EmHyperparameters,
}

/// Persistable result container for downstream credit assignment use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmFitResult {
    pub per_node: Vec<EmNodePosterior>,
    pub global: Vec<EmModePosterior>,
    pub config: EmGlobalConfigSnapshot,
    pub diagnostics: EmFitDiagnostics,
}

/// EM fitter boundary object (optimizer implementation to be added).
#[derive(Debug, Clone)]
pub struct EmFitter {
    pub hyperparameters: EmHyperparameters,
}
