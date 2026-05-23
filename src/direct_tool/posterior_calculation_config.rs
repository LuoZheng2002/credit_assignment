use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PosteriorCalculationConfig {
    pub temperature_to_accuracy: Vec<TemperatureAccuracyPair>,
    pub hyperparameters: PosteriorHyperparameters,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemperatureAccuracyPair {
    pub temperature: NotNan<f32>,
    pub accuracy: NotNan<f32>,
}

pub const DEFAULT_SIGMA_MEAN: f64 = 1.0;
pub const DEFAULT_SIGMA_LOG_STD: f64 = 1.0;
pub const DEFAULT_EPS: f64 = 1e-6;
pub const DEFAULT_MAX_ITERATIONS: usize = 120;
pub const DEFAULT_LOG_STD_CLAMP_MIN: f64 = -4.0;
pub const DEFAULT_LOG_STD_CLAMP_MAX: f64 = 2.0;
pub const DEFAULT_PRIOR_SCALE: f64 = 1.0;
pub const DEFAULT_PRIOR_CLIP_DELTA: f64 = 1e-6;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PosteriorHyperparameters {
    pub sigma_mean: NotNan<f64>,
    pub sigma_log_std: NotNan<f64>,
    pub eps: NotNan<f64>,
    pub max_iterations: usize,
    pub log_std_clamp_min: NotNan<f64>,
    pub log_std_clamp_max: NotNan<f64>,
    pub prior_scale: NotNan<f64>,
    pub prior_clip_delta: NotNan<f64>,
}

impl Default for PosteriorHyperparameters {
    fn default() -> Self {
        Self {
            sigma_mean: NotNan::new(DEFAULT_SIGMA_MEAN).unwrap(),
            sigma_log_std: NotNan::new(DEFAULT_SIGMA_LOG_STD).unwrap(),
            eps: NotNan::new(DEFAULT_EPS).unwrap(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            log_std_clamp_min: NotNan::new(DEFAULT_LOG_STD_CLAMP_MIN).unwrap(),
            log_std_clamp_max: NotNan::new(DEFAULT_LOG_STD_CLAMP_MAX).unwrap(),
            prior_scale: NotNan::new(DEFAULT_PRIOR_SCALE).unwrap(),
            prior_clip_delta: NotNan::new(DEFAULT_PRIOR_CLIP_DELTA).unwrap(),
        }
    }
}
