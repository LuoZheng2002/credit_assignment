use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DirectRolloutConfig {
    pub max_num_trunks: usize,
    pub max_num_total_trajectories: usize,
    pub temperature_fixed: bool,
    pub use_tool: bool,
}

