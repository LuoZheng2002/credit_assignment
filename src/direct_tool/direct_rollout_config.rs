use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DirectRolloutConfig {
    pub max_num_trunks: usize,
    pub max_num_total_trajectories: usize,
    pub fixed_temperature: NotNan<f32>,
    pub accuracy_under_temperature: Option<NotNan<f32>>,
    pub use_tool: bool,
}
