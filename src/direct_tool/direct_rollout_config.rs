use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DirectRolloutConfig {
    pub max_num_trunks: usize,
    pub max_num_total_trajectories: usize,
    pub temperature_fixed: bool,
    pub use_tool: bool,
}

impl DirectRolloutConfig {
    pub fn to_short_hash(&self) -> String {
        let serialized = serde_json::to_vec(self).unwrap();
        let hash = blake3::hash(&serialized);
        let short_hash = hex::encode(&hash.as_bytes()[..4]); // Take the first 4 bytes for a shorter hash
        assert_eq!(short_hash.len(), 8); // 4 bytes should give us 8 hex characters
        short_hash
    }
}
