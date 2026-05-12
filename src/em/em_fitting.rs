use crate::agent::trajectory_action_types::NodeType;

use super::em_types::{
    EmConstraintDiagnostics, EmFitDataset, EmFitDiagnostics, EmFitResult, EmGlobalConfigSnapshot,
    EmHyperparameters, EmLeafPath, EmLeafSlack, EmNodePosterior, EmNodeTypePosterior,
};

#[derive(Debug, Clone)]
pub struct EmFitter {
    pub hyperparameters: EmHyperparameters,
}

#[derive(Debug, Clone)]
struct EmIndexing {
    /// Ordered list of special node types that get free `mu_k` parameters.
    special_node_types: Vec<NodeType>,
    /// Data-calibrated prior mean for ordinary nodes (`NodeType::VerifierOff`).
    ordinary_prior_mean: f64,
}

#[derive(Debug, Clone)]
struct EmOptimizationState {
    node_means: Vec<f64>,
    node_log_stds: Vec<f64>,
    special_mus: Vec<f64>,
}

#[derive(Debug, Clone)]
struct EmObjectiveEvaluation {
    objective: f64,
    grad_node_means: Vec<f64>,
    grad_node_log_stds: Vec<f64>,
    grad_special_mus: Vec<f64>,
    per_leaf_slack: Vec<f64>,
}

#[derive(Debug, Clone)]
struct EmOptimizationOutcome {
    objective_trace: Vec<f64>,
    per_leaf_slack: Vec<f64>,
    converged: bool,
}

impl EmFitter {
    pub fn new(hyperparameters: EmHyperparameters) -> Self {
        Self { hyperparameters }
    }

    pub fn fit(&self, dataset: &EmFitDataset) -> EmFitResult {
        self.assert_hyperparameters();
        self.assert_dataset_integrity(dataset);

        let indexing = self.build_indexing(dataset);
        let mut state = self.initialize_state(dataset, &indexing);
        let optimization = self.run_optimizer(dataset, &indexing, &mut state);
        self.clamp_log_stds_in_place(&mut state.node_log_stds);

        self.assemble_result(dataset, &indexing, state, optimization)
    }

    fn assert_hyperparameters(&self) {
        assert!(
            self.hyperparameters.sigma_ordinary.is_finite()
                && self.hyperparameters.sigma_ordinary > 0.0,
            "sigma_ordinary must be finite and > 0"
        );
        assert!(
            self.hyperparameters.sigma_special.is_finite()
                && self.hyperparameters.sigma_special > 0.0,
            "sigma_special must be finite and > 0"
        );
        assert!(
            self.hyperparameters.sigma_log_std.is_finite()
                && self.hyperparameters.sigma_log_std > 0.0,
            "sigma_log_std must be finite and > 0"
        );
        assert!(
            self.hyperparameters.lambda_slack.is_finite(),
            "lambda_slack must be finite"
        );
        assert!(
            self.hyperparameters.eps.is_finite() && self.hyperparameters.eps >= 1e-8,
            "eps must be finite and >= 1e-8"
        );
        assert!(
            self.hyperparameters.max_iterations > 0,
            "max_iterations must be > 0"
        );
        assert!(
            self.hyperparameters.log_std_clamp.min.is_finite()
                && self.hyperparameters.log_std_clamp.max.is_finite()
                && self.hyperparameters.log_std_clamp.min < self.hyperparameters.log_std_clamp.max,
            "log_std_clamp bounds must be finite and min < max"
        );
    }

    fn assert_dataset_integrity(&self, dataset: &EmFitDataset) {
        assert!(
            !dataset.node_bindings.is_empty(),
            "EM fit dataset requires at least one node binding"
        );
        assert!(
            !dataset.leaf_bindings.is_empty(),
            "EM fit dataset requires at least one leaf binding"
        );
        assert_eq!(
            dataset.leaf_bindings.len(),
            dataset.leaf_paths.len(),
            "EM fit dataset requires one sparse path per leaf"
        );

        for (expected_leaf_id, leaf) in dataset.leaf_bindings.iter().enumerate() {
            assert_eq!(
                leaf.global_leaf_id, expected_leaf_id,
                "leaf_bindings must be aligned by global_leaf_id"
            );
        }
        for (expected_leaf_id, leaf_path) in dataset.leaf_paths.iter().enumerate() {
            assert_eq!(
                leaf_path.global_leaf_id, expected_leaf_id,
                "leaf_paths must be aligned by global_leaf_id"
            );
            assert!(
                !leaf_path.terms.is_empty(),
                "Each leaf path must contain at least one sparse path term"
            );
            for term in &leaf_path.terms {
                assert!(
                    term.global_node_id < dataset.node_bindings.len(),
                    "Sparse path term global_node_id out of bounds"
                );
                assert!(term.x_li.is_finite(), "Sparse path x_li must be finite");
            }
        }
    }

    fn build_indexing(&self, dataset: &EmFitDataset) -> EmIndexing {
        let mut special_node_types: Vec<NodeType> = Vec::new();
        for binding in &dataset.node_bindings {
            if !matches!(binding.node_type, NodeType::VerifierOff)
                && !special_node_types
                    .iter()
                    .any(|node_type| node_type_eq(node_type, &binding.node_type))
            {
                special_node_types.push(binding.node_type.clone());
            }
        }
        let ordinary_prior_mean = self.calibrated_ordinary_prior_mean(dataset);
        EmIndexing {
            special_node_types,
            ordinary_prior_mean,
        }
    }

    fn initialize_state(
        &self,
        dataset: &EmFitDataset,
        indexing: &EmIndexing,
    ) -> EmOptimizationState {
        let mut node_means: Vec<f64> = Vec::with_capacity(dataset.node_bindings.len());
        for binding in &dataset.node_bindings {
            let initial_mean = if matches!(binding.node_type, NodeType::VerifierOff) {
                indexing.ordinary_prior_mean
            } else {
                0.0
            };
            node_means.push(initial_mean);
        }

        EmOptimizationState {
            node_means,
            node_log_stds: vec![0.0; dataset.node_bindings.len()],
            special_mus: vec![0.0; indexing.special_node_types.len()],
        }
    }

    fn run_optimizer(
        &self,
        dataset: &EmFitDataset,
        indexing: &EmIndexing,
        state: &mut EmOptimizationState,
    ) -> EmOptimizationOutcome {
        let mut objective_trace: Vec<f64> = Vec::with_capacity(self.hyperparameters.max_iterations);
        for iteration in 0..self.hyperparameters.max_iterations {
            let eval = self.evaluate_objective_and_gradients(dataset, indexing, state);
            objective_trace.push(eval.objective);

            if iteration + 1 == self.hyperparameters.max_iterations {
                return EmOptimizationOutcome {
                    objective_trace,
                    per_leaf_slack: eval.per_leaf_slack,
                    converged: false,
                };
            }

            let grad_norm_sq = l2_norm_sq(&eval.grad_node_means)
                + l2_norm_sq(&eval.grad_node_log_stds)
                + l2_norm_sq(&eval.grad_special_mus);
            if grad_norm_sq <= 1e-18 {
                return EmOptimizationOutcome {
                    objective_trace,
                    per_leaf_slack: eval.per_leaf_slack,
                    converged: true,
                };
            }

            let mut step_size = 0.1;
            let mut accepted = false;
            for _ in 0..20 {
                let mut candidate = state.clone();
                gradient_step_in_place(
                    &mut candidate,
                    &eval.grad_node_means,
                    &eval.grad_node_log_stds,
                    &eval.grad_special_mus,
                    step_size,
                );
                self.clamp_log_stds_in_place(&mut candidate.node_log_stds);

                let candidate_eval =
                    self.evaluate_objective_and_gradients(dataset, indexing, &candidate);
                if candidate_eval.objective.is_finite() && candidate_eval.objective < eval.objective
                {
                    *state = candidate;
                    accepted = true;
                    break;
                }
                step_size *= 0.5;
            }

            if !accepted {
                return EmOptimizationOutcome {
                    objective_trace,
                    per_leaf_slack: eval.per_leaf_slack,
                    converged: false,
                };
            }
        }

        panic!("run_optimizer exited loop unexpectedly")
    }

    fn evaluate_objective_and_gradients(
        &self,
        dataset: &EmFitDataset,
        indexing: &EmIndexing,
        state: &EmOptimizationState,
    ) -> EmObjectiveEvaluation {
        assert_eq!(
            state.node_means.len(),
            dataset.node_bindings.len(),
            "node_means must align with node_bindings"
        );
        assert_eq!(
            state.node_log_stds.len(),
            dataset.node_bindings.len(),
            "node_log_stds must align with node_bindings"
        );
        assert_eq!(
            state.special_mus.len(),
            indexing.special_node_types.len(),
            "special_mus must align with special node types"
        );

        let inv_sigma_ordinary_sq =
            1.0 / (self.hyperparameters.sigma_ordinary * self.hyperparameters.sigma_ordinary);
        let inv_sigma_special_sq =
            1.0 / (self.hyperparameters.sigma_special * self.hyperparameters.sigma_special);
        let inv_sigma_log_std_sq =
            1.0 / (self.hyperparameters.sigma_log_std * self.hyperparameters.sigma_log_std);

        let mut objective = 0.0;
        let mut grad_node_means = vec![0.0; dataset.node_bindings.len()];
        let mut grad_node_log_stds = vec![0.0; dataset.node_bindings.len()];
        let mut grad_special_mus = vec![0.0; indexing.special_node_types.len()];

        for (node_idx, binding) in dataset.node_bindings.iter().enumerate() {
            let target_mean =
                self.node_type_prior_mean(indexing, &binding.node_type, &state.special_mus);
            let residual = state.node_means[node_idx] - target_mean;
            objective += 0.5 * inv_sigma_ordinary_sq * residual * residual;
            grad_node_means[node_idx] += inv_sigma_ordinary_sq * residual;

            if let Some(special_idx) = self.special_mu_index(indexing, &binding.node_type) {
                grad_special_mus[special_idx] += -inv_sigma_ordinary_sq * residual;
            }

            let u_i = state.node_log_stds[node_idx];
            objective += 0.5 * inv_sigma_log_std_sq * u_i * u_i;
            grad_node_log_stds[node_idx] += inv_sigma_log_std_sq * u_i;
        }

        for (special_idx, mu_k) in state.special_mus.iter().copied().enumerate() {
            objective += 0.5 * inv_sigma_special_sq * mu_k * mu_k;
            grad_special_mus[special_idx] += inv_sigma_special_sq * mu_k;
        }

        let exp_two_u: Vec<f64> = state
            .node_log_stds
            .iter()
            .copied()
            .map(|u_i| {
                let v = (2.0 * u_i).exp();
                assert!(
                    v.is_finite() && v > 0.0,
                    "exp(2*u_i) must be finite and > 0"
                );
                v
            })
            .collect();

        let mut per_leaf_slack: Vec<f64> = Vec::with_capacity(dataset.leaf_bindings.len());
        for (leaf_idx, leaf_binding) in dataset.leaf_bindings.iter().enumerate() {
            let leaf_path = &dataset.leaf_paths[leaf_idx];
            assert_eq!(
                leaf_binding.global_leaf_id, leaf_path.global_leaf_id,
                "leaf binding and leaf path must align on global_leaf_id"
            );

            let (mu_l, var_l) = self.compute_leaf_score_stats(leaf_path, state, &exp_two_u);
            let y_l = leaf_binding.label.as_sign();
            let tau_l = (var_l + self.hyperparameters.eps).sqrt();
            assert!(
                tau_l.is_finite() && tau_l > 0.0,
                "leaf tau must be finite and > 0"
            );

            let z_l = y_l * mu_l / tau_l;
            let xi_l = (-z_l).max(0.0);
            per_leaf_slack.push(xi_l);

            let (log_cdf, pdf_over_cdf) = stable_log_standard_normal_cdf_and_pdf_over_cdf(z_l);
            objective += -log_cdf;

            let d_obj_d_z = -pdf_over_cdf;
            let tau_cubed = tau_l * tau_l * tau_l;
            assert!(
                tau_cubed.is_finite() && tau_cubed > 0.0,
                "tau^3 must be finite and > 0"
            );
            for term in &leaf_path.terms {
                let node_idx = term.global_node_id;
                let x_li = term.x_li;
                grad_node_means[node_idx] += d_obj_d_z * y_l * x_li / tau_l;

                let d_z_d_u_i = -y_l * mu_l * x_li * exp_two_u[node_idx] / tau_cubed;
                grad_node_log_stds[node_idx] += d_obj_d_z * d_z_d_u_i;
            }
        }

        EmObjectiveEvaluation {
            objective,
            grad_node_means,
            grad_node_log_stds,
            grad_special_mus,
            per_leaf_slack,
        }
    }

    fn clamp_log_stds_in_place(&self, node_log_stds: &mut [f64]) {
        for u_i in node_log_stds {
            *u_i = u_i.clamp(
                self.hyperparameters.log_std_clamp.min,
                self.hyperparameters.log_std_clamp.max,
            );
        }
    }

    fn assemble_result(
        &self,
        dataset: &EmFitDataset,
        indexing: &EmIndexing,
        state: EmOptimizationState,
        optimization: EmOptimizationOutcome,
    ) -> EmFitResult {
        let per_node: Vec<EmNodePosterior> = dataset
            .node_bindings
            .iter()
            .enumerate()
            .map(|(node_idx, binding)| EmNodePosterior {
                global_node_id: binding.global_node_id,
                tree_question_id: binding.tree_question_id,
                node_id: binding.node_id,
                mean: state.node_means[node_idx],
                log_std: state.node_log_stds[node_idx],
                node_type: binding.node_type.clone(),
            })
            .collect();

        let global = self.build_global_posteriors(indexing, &state);
        let constraints = self.build_constraint_diagnostics(dataset, &optimization.per_leaf_slack);
        let final_train_sign_accuracy = self.compute_train_sign_accuracy(dataset, &state);
        let mean_slack_train = if optimization.per_leaf_slack.is_empty() {
            0.0
        } else {
            optimization.per_leaf_slack.iter().sum::<f64>()
                / optimization.per_leaf_slack.len() as f64
        };

        EmFitResult {
            per_node,
            global,
            config: EmGlobalConfigSnapshot {
                hyperparameters: self.hyperparameters.clone(),
                ordinary_prior_mean: indexing.ordinary_prior_mean,
            },
            diagnostics: EmFitDiagnostics {
                objective_trace: optimization.objective_trace,
                converged_flag: optimization.converged,
                final_train_sign_accuracy,
                final_val_sign_accuracy: None,
                mean_slack_train,
                mean_slack_val: None,
                constraints,
            },
        }
    }

    fn build_global_posteriors(
        &self,
        indexing: &EmIndexing,
        state: &EmOptimizationState,
    ) -> Vec<EmNodeTypePosterior> {
        assert_eq!(
            indexing.special_node_types.len(),
            state.special_mus.len(),
            "special node types and special mus must have identical lengths"
        );
        indexing
            .special_node_types
            .iter()
            .cloned()
            .zip(state.special_mus.iter().copied())
            .map(|(node_type, mu_k)| EmNodeTypePosterior { node_type, mu_k })
            .collect()
    }

    fn build_constraint_diagnostics(
        &self,
        dataset: &EmFitDataset,
        per_leaf_slack: &[f64],
    ) -> EmConstraintDiagnostics {
        assert_eq!(
            dataset.leaf_bindings.len(),
            per_leaf_slack.len(),
            "per_leaf_slack must align with leaf_bindings"
        );

        let sum_xi = per_leaf_slack.iter().sum::<f64>();
        let num_positive_xi = per_leaf_slack.iter().filter(|&&xi| xi > 0.0).count();

        let mut with_index: Vec<(usize, f64)> =
            per_leaf_slack.iter().copied().enumerate().collect();
        with_index.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("slack values must be comparable")
        });

        let largest_violators: Vec<EmLeafSlack> = with_index
            .into_iter()
            .take(10)
            .map(|(leaf_idx, slack)| {
                let leaf_binding = &dataset.leaf_bindings[leaf_idx];
                EmLeafSlack {
                    global_leaf_id: leaf_binding.global_leaf_id,
                    tree_question_id: leaf_binding.tree_question_id,
                    leaf_node_id: leaf_binding.leaf_node_id,
                    slack,
                }
            })
            .collect();

        EmConstraintDiagnostics {
            sum_xi,
            num_positive_xi,
            largest_violators,
        }
    }

    fn compute_train_sign_accuracy(
        &self,
        dataset: &EmFitDataset,
        state: &EmOptimizationState,
    ) -> f64 {
        let exp_two_u: Vec<f64> = state
            .node_log_stds
            .iter()
            .copied()
            .map(|u_i| {
                let v = (2.0 * u_i).exp();
                assert!(
                    v.is_finite() && v > 0.0,
                    "exp(2*u_i) must be finite and > 0"
                );
                v
            })
            .collect();

        let mut num_correct = 0usize;
        for (leaf_idx, leaf_binding) in dataset.leaf_bindings.iter().enumerate() {
            let leaf_path = &dataset.leaf_paths[leaf_idx];
            let (mu_l, var_l) = self.compute_leaf_score_stats(leaf_path, state, &exp_two_u);
            let tau_l = (var_l + self.hyperparameters.eps).sqrt();
            assert!(
                tau_l.is_finite() && tau_l > 0.0,
                "leaf tau must be finite and > 0"
            );
            let normalized_margin = leaf_binding.label.as_sign() * mu_l / tau_l;
            if normalized_margin >= 0.0 {
                num_correct += 1;
            }
        }
        num_correct as f64 / dataset.leaf_bindings.len() as f64
    }

    fn compute_leaf_score_stats(
        &self,
        leaf_path: &EmLeafPath,
        state: &EmOptimizationState,
        exp_two_u: &[f64],
    ) -> (f64, f64) {
        assert_eq!(
            state.node_means.len(),
            exp_two_u.len(),
            "node_means and exp_two_u must have identical lengths"
        );

        let mut mu_l = 0.0;
        let mut var_l = 0.0;
        for term in &leaf_path.terms {
            assert!(
                term.global_node_id < state.node_means.len(),
                "leaf path references global_node_id out of bounds"
            );
            assert!(
                term.x_li.is_finite() && term.x_li >= 0.0,
                "x_li must be finite and >= 0"
            );
            let node_idx = term.global_node_id;
            let x_li = term.x_li;
            mu_l += x_li * state.node_means[node_idx];
            var_l += x_li * exp_two_u[node_idx];
        }

        assert!(
            var_l.is_finite() && var_l > 0.0,
            "leaf variance must be finite and > 0"
        );
        (mu_l, var_l)
    }

    fn node_type_prior_mean(
        &self,
        indexing: &EmIndexing,
        node_type: &NodeType,
        special_mus: &[f64],
    ) -> f64 {
        if let Some(special_idx) = self.special_mu_index(indexing, node_type) {
            return indexing.ordinary_prior_mean + special_mus[special_idx];
        }
        indexing.ordinary_prior_mean
    }

    fn calibrated_ordinary_prior_mean(&self, dataset: &EmFitDataset) -> f64 {
        assert!(
            !dataset.leaf_bindings.is_empty(),
            "EM fit dataset requires at least one leaf binding"
        );
        assert_eq!(
            dataset.leaf_bindings.len(),
            dataset.leaf_paths.len(),
            "EM fit dataset requires one sparse path per leaf"
        );

        let num_correct = dataset
            .leaf_bindings
            .iter()
            .filter(|leaf| leaf.label.as_sign() > 0.0)
            .count();
        let empirical_accuracy = num_correct as f64 / dataset.leaf_bindings.len() as f64;
        assert!(
            empirical_accuracy.is_finite() && (0.0..=1.0).contains(&empirical_accuracy),
            "empirical accuracy must be finite and within [0, 1]"
        );

        let p = empirical_accuracy.clamp(1e-6, 1.0 - 1e-6);
        let baseline_margin = standard_normal_inverse_cdf(p);

        let mut sum_sqrt_path_mass = 0.0;
        for leaf_path in &dataset.leaf_paths {
            let path_mass = leaf_path.terms.iter().map(|term| term.x_li).sum::<f64>();
            assert!(
                path_mass.is_finite() && path_mass > 0.0,
                "Each leaf path must have positive finite path mass"
            );
            sum_sqrt_path_mass += path_mass.sqrt();
        }
        let mean_sqrt_path_mass = sum_sqrt_path_mass / dataset.leaf_paths.len() as f64;
        assert!(
            mean_sqrt_path_mass.is_finite() && mean_sqrt_path_mass > 0.0,
            "Mean sqrt path mass must be finite and > 0"
        );

        baseline_margin / mean_sqrt_path_mass
    }

    fn special_mu_index(&self, indexing: &EmIndexing, node_type: &NodeType) -> Option<usize> {
        if matches!(node_type, NodeType::VerifierOff) {
            return None;
        }
        indexing
            .special_node_types
            .iter()
            .position(|candidate| node_type_eq(candidate, node_type))
            .or_else(|| {
                panic!(
                    "Missing special node type in indexing for non-ordinary node: {:?}",
                    node_type
                )
            })
    }
}

fn l2_norm_sq(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>()
}

fn gradient_step_in_place(
    state: &mut EmOptimizationState,
    grad_node_means: &[f64],
    grad_node_log_stds: &[f64],
    grad_special_mus: &[f64],
    step_size: f64,
) {
    assert!(
        step_size.is_finite() && step_size > 0.0,
        "step_size must be finite and > 0"
    );
    assert_eq!(
        state.node_means.len(),
        grad_node_means.len(),
        "grad_node_means must align with node_means"
    );
    assert_eq!(
        state.node_log_stds.len(),
        grad_node_log_stds.len(),
        "grad_node_log_stds must align with node_log_stds"
    );
    assert_eq!(
        state.special_mus.len(),
        grad_special_mus.len(),
        "grad_special_mus must align with special_mus"
    );

    for (value, grad) in state
        .node_means
        .iter_mut()
        .zip(grad_node_means.iter().copied())
    {
        *value -= step_size * grad;
    }
    for (value, grad) in state
        .node_log_stds
        .iter_mut()
        .zip(grad_node_log_stds.iter().copied())
    {
        *value -= step_size * grad;
    }
    for (value, grad) in state
        .special_mus
        .iter_mut()
        .zip(grad_special_mus.iter().copied())
    {
        *value -= step_size * grad;
    }
}

fn node_type_eq(lhs: &NodeType, rhs: &NodeType) -> bool {
    matches!(
        (lhs, rhs),
        (NodeType::VerifierOff, NodeType::VerifierOff)
            | (NodeType::VerifierOn, NodeType::VerifierOn)
            | (
                NodeType::VerifierOnAndOverwriteLastStep,
                NodeType::VerifierOnAndOverwriteLastStep
            )
            | (
                NodeType::VerifierOnAndChangePlan,
                NodeType::VerifierOnAndChangePlan
            )
    )
}

fn stable_log_standard_normal_cdf_and_pdf_over_cdf(z: f64) -> (f64, f64) {
    assert!(z.is_finite(), "z must be finite");

    if z > -10.0 {
        let cdf = standard_normal_cdf(z).max(1e-300);
        let pdf = standard_normal_pdf(z);
        let log_cdf = cdf.ln();
        let pdf_over_cdf = (pdf / cdf).max(0.0);
        assert!(log_cdf.is_finite(), "log_cdf must be finite");
        assert!(pdf_over_cdf.is_finite(), "pdf_over_cdf must be finite");
        return (log_cdf, pdf_over_cdf);
    }

    let x = -z;
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    let inv_x4 = inv_x2 * inv_x2;
    let series = 1.0 - inv_x2 + 3.0 * inv_x4;
    assert!(
        series.is_finite() && series > 0.0,
        "tail series must be finite and > 0"
    );

    let log_cdf = -0.5 * z * z - x.ln() - LOG_SQRT_2PI + series.ln();
    let pdf_over_cdf = x + inv_x - 2.0 * inv_x * inv_x2;
    assert!(log_cdf.is_finite(), "tail log_cdf must be finite");
    assert!(
        pdf_over_cdf.is_finite() && pdf_over_cdf > 0.0,
        "tail pdf_over_cdf must be finite and > 0"
    );
    (log_cdf, pdf_over_cdf)
}

fn standard_normal_pdf(z: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z * FRAC_1_SQRT_2))
}

fn standard_normal_inverse_cdf(p: f64) -> f64 {
    assert!(
        p.is_finite() && p > 0.0 && p < 1.0,
        "p must be finite and in (0, 1)"
    );

    // Acklam's rational approximation for inverse normal CDF.
    const A1: f64 = -3.969_683_028_665_376e1;
    const A2: f64 = 2.209_460_984_245_205e2;
    const A3: f64 = -2.759_285_104_469_687e2;
    const A4: f64 = 1.383_577_518_672_69e2;
    const A5: f64 = -3.066_479_806_614_716e1;
    const A6: f64 = 2.506_628_277_459_239;
    const B1: f64 = -5.447_609_879_822_406e1;
    const B2: f64 = 1.615_858_368_580_409e2;
    const B3: f64 = -1.556_989_798_598_866e2;
    const B4: f64 = 6.680_131_188_771_972e1;
    const B5: f64 = -1.328_068_155_288_572e1;
    const C1: f64 = -7.784_894_002_430_293e-3;
    const C2: f64 = -3.223_964_580_411_365e-1;
    const C3: f64 = -2.400_758_277_161_838;
    const C4: f64 = -2.549_732_539_343_734;
    const C5: f64 = 4.374_664_141_464_968;
    const C6: f64 = 2.938_163_982_698_783;
    const D1: f64 = 7.784_695_709_041_462e-3;
    const D2: f64 = 3.224_671_290_700_398e-1;
    const D3: f64 = 2.445_134_137_142_996;
    const D4: f64 = 3.754_408_661_907_416;
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6;
        let den = (((D1 * q + D2) * q + D3) * q + D4) * q + 1.0;
        return num / den;
    }

    if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * q;
        let den = ((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0;
        return num / den;
    }

    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    let num = -(((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6);
    let den = (((D1 * q + D2) * q + D3) * q + D4) * q + 1.0;
    num / den
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.5 * a);
    let poly = -a * a - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    sign * (1.0 - t * poly.exp())
}

const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const FRAC_1_SQRT_2: f64 = 0.707_106_781_186_547_5;
