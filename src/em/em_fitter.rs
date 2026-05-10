use serde::{Deserialize, Serialize};

use crate::agent::trajectory_action_types::VerifierAndModeSummary;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeafLabel {
    /// Leaf trajectory judged as successful/correct (`y_l = +1`).
    Correct,
    /// Leaf trajectory judged as failed/incorrect (`y_l = -1`).
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
    /// Lower bound for per-node `u_i = log_std_i` during optimization.
    pub min: f64,
    /// Upper bound for per-node `u_i = log_std_i` during optimization.
    pub max: f64,
}

/// Hyperparameters for the deterministic-sign + slack EM objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmHyperparameters {
    /// Prior std in `m_i ~ N(mu_mode(i), sigma_mean^2)`.
    pub sigma_mean: f64,
    /// Prior std for shared mode centers (`mu_k` and `nu_k`).
    pub sigma_mode: f64,
    /// Prior std in `u_i ~ N(nu_mode(i), sigma_log_std^2)`.
    pub sigma_log_std: f64,
    /// Prior center for mode-level log-std means `nu_k`.
    pub mu_log_std_mode: f64,
    /// Coefficient for `sum_l xi_l^2` slack penalty.
    pub lambda_slack: f64,
    /// Numerical stabilizer used in normalized sign constraints.
    pub eps: f64,
    /// Fixed-iteration budget (current stopping rule).
    pub max_iterations: usize,
    /// Clamp range for per-node log-std values.
    pub log_std_clamp: LogStdClamp,
}

/// Mapping from flattened global node index to original tree/node identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmNodeBinding {
    pub global_node_id: usize,
    pub tree_question_id: usize,
    pub node_id: usize,
    /// Mode used for both contribution and log-std priors.
    pub mode: VerifierAndModeSummary,
}

/// Mapping from flattened global leaf index to original tree leaf and label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmLeafBinding {
    pub global_leaf_id: usize,
    pub tree_question_id: usize,
    pub leaf_node_id: usize,
    pub label: LeafLabel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsePathTerm {
    /// Referenced node index in global node space.
    pub global_node_id: usize,
    /// Sparse path indicator (`x_{l,i}`), currently expected to be 1.0.
    pub x_li: f64,
}

/// Sparse representation of one judged leaf path over global nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmLeafPath {
    pub global_leaf_id: usize,
    /// Non-zero terms for this leaf row in the path matrix.
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
    /// Full node catalog in global index order.
    pub node_bindings: Vec<EmNodeBinding>,
    /// Full judged-leaf catalog in global index order.
    pub leaf_bindings: Vec<EmLeafBinding>,
    /// Sparse path rows aligned by `global_leaf_id`.
    pub leaf_paths: Vec<EmLeafPath>,
}

/// Per-node fitted posterior parameters used by downstream credit assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmNodePosterior {
    pub global_node_id: usize,
    pub tree_question_id: usize,
    pub node_id: usize,
    /// Fitted contribution mean (`m_i`).
    pub mean: f64,
    /// Fitted log standard deviation (`u_i = log_std_i`).
    pub log_std: f64,
    pub mode: VerifierAndModeSummary,
}

/// Fitted shared priors for each mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmModePosterior {
    pub mode: VerifierAndModeSummary,
    /// Contribution prior center for this mode.
    pub mu_k: f64,
    /// Log-std prior center for this mode.
    pub nu_k: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmConstraintDiagnostics {
    /// Aggregate slack magnitude over all judged leaves.
    pub sum_xi: f64,
    /// Number of leaves with strictly positive slack.
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
    /// Objective value per optimizer iteration.
    pub objective_trace: Vec<f64>,
    /// Reserved for future tolerance-based stopping; currently usually false.
    pub converged_flag: bool,
    /// Fraction of leaves satisfying the sign decision after fit.
    pub final_train_sign_accuracy: f64,
    /// Validation is deferred for now; kept for schema compatibility.
    pub final_val_sign_accuracy: Option<f64>,
    pub mean_slack_train: f64,
    /// Validation is deferred for now; kept for schema compatibility.
    pub mean_slack_val: Option<f64>,
    pub constraints: EmConstraintDiagnostics,
}

/// Serializable snapshot of fitting-time global configuration.
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
