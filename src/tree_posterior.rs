use std::collections::BTreeMap;

use crate::{
    hybrid_dataset::DatasetSplit,
    llm_model::LlmModelMarker,
    posterior_calculation_config::PosteriorHyperparameters,
    tree::{DirectTree, SegmentId},
};

#[derive(Debug, Clone)]
pub struct Posterior {
    pub mean: f32,
    pub log_std: f32,
}

#[derive(Debug, Clone)]
struct LeafPathObservation {
    y_label_sign: f64,
    segment_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PosteriorOptimizationState {
    means: Vec<f64>,
    log_stds: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PosteriorObjectiveEvaluation {
    objective: f64,
    grad_means: Vec<f64>,
    grad_log_stds: Vec<f64>,
}

impl<'a, M: LlmModelMarker, S: DatasetSplit> DirectTree<'a, M, S> {
    pub fn calculate_segment_posteriors(
        &self,
        override_hyperparameters: Option<&PosteriorHyperparameters>,
    ) -> BTreeMap<SegmentId, Posterior> {
        if self.segments.is_empty() {
            return BTreeMap::new();
        }

        let root_segment_id = self
            .root_segment_id
            .expect("DirectTree must have root_segment_id when segments are non-empty");

        let segment_ids: Vec<SegmentId> = self
            .segments
            .keys()
            .copied()
            .filter(|segment_id| *segment_id != root_segment_id)
            .collect();
        if segment_ids.is_empty() {
            return BTreeMap::new();
        }

        assert!(
            !self.leaf_segment_judgments.is_empty(),
            "Cannot calculate posteriors without leaf judgments"
        );

        let hyperparameters = override_hyperparameters
            .unwrap_or(&self.action_log.posterior_calculation_config.hyperparameters);
        let segment_index: BTreeMap<SegmentId, usize> = segment_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, segment_id)| (segment_id, idx))
            .collect();
        let leaf_paths = self.build_leaf_paths(&segment_index);

        let prior_means_per_segment = vec![0.0; segment_ids.len()];
        let mut state = PosteriorOptimizationState {
            means: prior_means_per_segment.clone(),
            log_stds: vec![0.0; segment_ids.len()],
        };

        self.optimize_posteriors(
            &leaf_paths,
            &prior_means_per_segment,
            &hyperparameters,
            &mut state,
        );
        self.clamp_log_stds(&mut state.log_stds, &hyperparameters);

        segment_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, segment_id)| {
                (
                    segment_id,
                    Posterior {
                        mean: state.means[idx] as f32,
                        log_std: state.log_stds[idx] as f32,
                    },
                )
            })
            .collect()
    }

    fn build_leaf_paths(
        &self,
        segment_index: &BTreeMap<SegmentId, usize>,
    ) -> Vec<LeafPathObservation> {
        let mut leaf_paths: Vec<LeafPathObservation> =
            Vec::with_capacity(self.leaf_segment_judgments.len());
        for (leaf_segment_id, judgment) in &self.leaf_segment_judgments {
            let mut path_segment_indices: Vec<usize> = Vec::new();
            let mut current_segment_id = Some(*leaf_segment_id);
            while let Some(segment_id) = current_segment_id {
                if let Some(idx) = segment_index.get(&segment_id).copied() {
                    path_segment_indices.push(idx);
                }
                let segment = self
                    .segments
                    .get(&segment_id)
                    .expect("Traversed segment id must exist in tree");
                current_segment_id = segment.parent_id;
            }
            if path_segment_indices.is_empty() {
                continue;
            }

            let y_label_sign = if judgment.is_correct { 1.0 } else { -1.0 };
            leaf_paths.push(LeafPathObservation {
                y_label_sign,
                segment_indices: path_segment_indices,
            });
        }
        leaf_paths
    }

    fn optimize_posteriors(
        &self,
        leaf_paths: &[LeafPathObservation],
        prior_means_per_segment: &[f64],
        hyperparameters: &PosteriorHyperparameters,
        state: &mut PosteriorOptimizationState,
    ) {
        for _ in 0..hyperparameters.max_iterations {
            let eval = evaluate_objective_and_gradients(
                leaf_paths,
                prior_means_per_segment,
                hyperparameters,
                state,
            );

            let grad_norm_sq = l2_norm_sq(&eval.grad_means) + l2_norm_sq(&eval.grad_log_stds);
            if grad_norm_sq <= 1e-18 {
                break;
            }

            let mut step_size = 0.1;
            let mut accepted = false;
            for _ in 0..20 {
                let mut candidate = state.clone();
                gradient_step_in_place(
                    &mut candidate,
                    &eval.grad_means,
                    &eval.grad_log_stds,
                    step_size,
                );
                self.clamp_log_stds(&mut candidate.log_stds, hyperparameters);

                let candidate_eval = evaluate_objective_and_gradients(
                    leaf_paths,
                    prior_means_per_segment,
                    hyperparameters,
                    &candidate,
                );
                if candidate_eval.objective.is_finite() && candidate_eval.objective < eval.objective
                {
                    *state = candidate;
                    accepted = true;
                    break;
                }
                step_size *= 0.5;
            }

            if !accepted {
                break;
            }
        }
    }

    fn clamp_log_stds(&self, log_stds: &mut [f64], hyperparameters: &PosteriorHyperparameters) {
        for value in log_stds {
            *value = value.clamp(
                hyperparameters.log_std_clamp_min.into_inner(),
                hyperparameters.log_std_clamp_max.into_inner(),
            );
        }
    }
}

fn evaluate_objective_and_gradients(
    leaf_paths: &[LeafPathObservation],
    prior_means_per_segment: &[f64],
    hyperparameters: &PosteriorHyperparameters,
    state: &PosteriorOptimizationState,
) -> PosteriorObjectiveEvaluation {
    assert_eq!(
        prior_means_per_segment.len(),
        state.means.len(),
        "prior means and means must align"
    );
    assert_eq!(
        prior_means_per_segment.len(),
        state.log_stds.len(),
        "prior means and log stds must align"
    );

    let inv_sigma_mean_sq =
        1.0 / (hyperparameters.sigma_mean.into_inner() * hyperparameters.sigma_mean.into_inner());
    let inv_sigma_log_std_sq = 1.0
        / (hyperparameters.sigma_log_std.into_inner() * hyperparameters.sigma_log_std.into_inner());

    let mut objective = 0.0;
    let mut grad_means = vec![0.0; state.means.len()];
    let mut grad_log_stds = vec![0.0; state.log_stds.len()];

    for i in 0..state.means.len() {
        let residual = state.means[i] - prior_means_per_segment[i];
        objective += 0.5 * inv_sigma_mean_sq * residual * residual;
        grad_means[i] += inv_sigma_mean_sq * residual;

        let u = state.log_stds[i];
        objective += 0.5 * inv_sigma_log_std_sq * u * u;
        grad_log_stds[i] += inv_sigma_log_std_sq * u;
    }

    let exp_two_u: Vec<f64> = state
        .log_stds
        .iter()
        .copied()
        .map(|u| {
            let v = (2.0 * u).exp();
            assert!(v.is_finite() && v > 0.0, "exp(2u) must be finite and > 0");
            v
        })
        .collect();

    for leaf in leaf_paths {
        let mut mu_l = 0.0;
        let mut var_l = 0.0;
        for &segment_idx in &leaf.segment_indices {
            mu_l += state.means[segment_idx];
            var_l += exp_two_u[segment_idx];
        }

        let tau_l = (var_l + hyperparameters.eps.into_inner()).sqrt();
        assert!(
            tau_l.is_finite() && tau_l > 0.0,
            "leaf tau must be finite and > 0"
        );

        let z_l = leaf.y_label_sign * mu_l / tau_l;
        let (log_cdf, pdf_over_cdf) = stable_log_standard_normal_cdf_and_pdf_over_cdf(z_l);
        objective += -log_cdf;

        let d_obj_d_z = -pdf_over_cdf;
        let tau_cubed = tau_l * tau_l * tau_l;
        assert!(
            tau_cubed.is_finite() && tau_cubed > 0.0,
            "tau cubed must be finite and > 0"
        );

        for &segment_idx in &leaf.segment_indices {
            grad_means[segment_idx] += d_obj_d_z * leaf.y_label_sign / tau_l;

            let d_z_d_u_i = -leaf.y_label_sign * mu_l * exp_two_u[segment_idx] / tau_cubed;
            grad_log_stds[segment_idx] += d_obj_d_z * d_z_d_u_i;
        }
    }

    PosteriorObjectiveEvaluation {
        objective,
        grad_means,
        grad_log_stds,
    }
}

fn gradient_step_in_place(
    state: &mut PosteriorOptimizationState,
    grad_means: &[f64],
    grad_log_stds: &[f64],
    step_size: f64,
) {
    assert!(
        step_size.is_finite() && step_size > 0.0,
        "step size must be > 0"
    );
    assert_eq!(
        state.means.len(),
        grad_means.len(),
        "gradient and state lengths must align for means"
    );
    assert_eq!(
        state.log_stds.len(),
        grad_log_stds.len(),
        "gradient and state lengths must align for log stds"
    );

    for (value, grad) in state.means.iter_mut().zip(grad_means.iter().copied()) {
        *value -= step_size * grad;
    }
    for (value, grad) in state.log_stds.iter_mut().zip(grad_log_stds.iter().copied()) {
        *value -= step_size * grad;
    }
}

fn l2_norm_sq(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>()
}

fn stable_log_standard_normal_cdf_and_pdf_over_cdf(z: f64) -> (f64, f64) {
    assert!(z.is_finite(), "z must be finite");

    if z > -10.0 {
        let cdf = standard_normal_cdf(z).max(1e-300);
        let pdf = standard_normal_pdf(z);
        let log_cdf = cdf.ln();
        let pdf_over_cdf = (pdf / cdf).max(0.0);
        return (log_cdf, pdf_over_cdf);
    }

    let x = -z;
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    let inv_x4 = inv_x2 * inv_x2;
    let series = 1.0 - inv_x2 + 3.0 * inv_x4;
    assert!(
        series.is_finite() && series > 0.0,
        "tail series must be > 0"
    );

    let log_cdf = -0.5 * z * z - x.ln() - LOG_SQRT_2PI + series.ln();
    let pdf_over_cdf = x + inv_x - 2.0 * inv_x * inv_x2;
    assert!(
        pdf_over_cdf.is_finite() && pdf_over_cdf > 0.0,
        "tail pdf over cdf must be finite and > 0"
    );
    (log_cdf, pdf_over_cdf)
}

fn standard_normal_pdf(z: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z * FRAC_1_SQRT_2))
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
