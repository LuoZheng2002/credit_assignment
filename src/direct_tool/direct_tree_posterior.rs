use std::collections::BTreeMap;

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentId},
        posterior_calculation_config::{PosteriorHyperparameters, TemperatureAccuracyPair},
    },
    llm_model::LlmModelMarker,
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

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn calculate_segment_posteriors(&self) -> BTreeMap<SegmentId, Posterior> {
        if self.segments.is_empty() {
            return BTreeMap::new();
        }

        assert!(
            !self.leaf_segment_judgments.is_empty(),
            "Cannot calculate posteriors without leaf judgments"
        );

        let hyperparameters = self.posterior_calculation_config.hyperparameters;
        let segment_ids: Vec<SegmentId> = self.segments.keys().copied().collect();
        let segment_index: BTreeMap<SegmentId, usize> = segment_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, segment_id)| (segment_id, idx))
            .collect();
        let leaf_paths = self.build_leaf_paths(&segment_index);

        let prior_means_per_segment = self.temperature_conditioned_prior_means(&segment_ids);
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
                let idx = *segment_index
                    .get(&segment_id)
                    .expect("Every traversed segment must be indexable");
                path_segment_indices.push(idx);
                let segment = self
                    .segments
                    .get(&segment_id)
                    .expect("Traversed segment id must exist in tree");
                current_segment_id = segment.parent_id;
            }
            assert!(
                !path_segment_indices.is_empty(),
                "Each leaf path must contain at least one segment"
            );

            let y_label_sign = if judgment.is_correct { 1.0 } else { -1.0 };
            leaf_paths.push(LeafPathObservation {
                y_label_sign,
                segment_indices: path_segment_indices,
            });
        }
        leaf_paths
    }

    fn temperature_conditioned_prior_means(&self, segment_ids: &[SegmentId]) -> Vec<f64> {
        let hyperparameters = &self.posterior_calculation_config.hyperparameters;
        let temperature_to_accuracy = &self.posterior_calculation_config.temperature_to_accuracy;

        let global_accuracy = if temperature_to_accuracy.is_empty() {
            0.5
        } else {
            temperature_to_accuracy
                .iter()
                .map(
                    |TemperatureAccuracyPair {
                         temperature: _,
                         accuracy,
                     }| accuracy.into_inner() as f64,
                )
                .sum::<f64>()
                / temperature_to_accuracy.len() as f64
        };

        segment_ids
            .iter()
            .map(|segment_id| {
                let segment = self
                    .segments
                    .get(segment_id)
                    .expect("Segment id must exist while creating priors");
                let segment_temperature = segment.llm_temperature;
                let temperature_accuracy = interpolated_temperature_accuracy(
                    segment_temperature,
                    &temperature_to_accuracy,
                    global_accuracy,
                );
                let clipped = temperature_accuracy.clamp(
                    hyperparameters.prior_clip_delta.into_inner(),
                    1.0 - hyperparameters.prior_clip_delta.into_inner(),
                );
                hyperparameters.prior_scale.into_inner() * standard_normal_inverse_cdf(clipped)
            })
            .collect()
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

pub struct TemperatureAccuracyPairFloat {
    pub temperature: f32,
    pub accuracy: f32,
}

fn interpolated_temperature_accuracy(
    target_temperature: f32,
    grouped_accuracy: &[TemperatureAccuracyPair],
    default_accuracy: f64,
) -> f64 {
    if grouped_accuracy.is_empty() {
        return default_accuracy;
    }
    let mut pairs: Vec<TemperatureAccuracyPairFloat> = grouped_accuracy
        .iter()
        .map(
            |TemperatureAccuracyPair {
                 temperature,
                 accuracy,
             }| TemperatureAccuracyPairFloat {
                temperature: temperature.into_inner(),
                accuracy: accuracy.into_inner(),
            },
        )
        .collect();
    for TemperatureAccuracyPairFloat {
        temperature: _,
        accuracy,
    } in pairs.iter()
    {
        let accuracy = *accuracy;
        assert!(accuracy.is_finite(), "Temperature accuracy must be finite");
        assert!(
            (0.0..=1.0).contains(&accuracy),
            "Temperature accuracy must be in [0, 1]"
        );
    }
    pairs.sort_by(|lhs, rhs| {
        lhs.temperature
            .partial_cmp(&rhs.temperature)
            .expect("Temperature keys must be comparable")
    });

    let first_pair = &pairs[0];
    if target_temperature <= first_pair.temperature {
        return first_pair.accuracy as f64;
    }
    let last_pair = &pairs[pairs.len() - 1];
    if target_temperature >= last_pair.temperature {
        return last_pair.accuracy as f64;
    }

    for window in pairs.windows(2) {
        let left_pair = &window[0];
        let right_pair = &window[1];
        if target_temperature < left_pair.temperature || target_temperature > right_pair.temperature
        {
            continue;
        }
        let span = right_pair.temperature - left_pair.temperature;
        if span.abs() <= 1e-8 {
            return right_pair.accuracy as f64;
        }
        let ratio = (target_temperature - left_pair.temperature) / span;
        let interpolated = left_pair.accuracy + ratio * (right_pair.accuracy - left_pair.accuracy);
        return interpolated as f64;
    }
    default_accuracy
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

fn standard_normal_inverse_cdf(p: f64) -> f64 {
    assert!(
        p.is_finite() && p > 0.0 && p < 1.0,
        "p must be finite and in (0, 1)"
    );

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
