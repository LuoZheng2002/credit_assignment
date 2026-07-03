use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use crate::direct_tool::hybrid_dataset::DatasetSplit;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum BranchingPolicy {
    TreeMappoGuided, // our method with guided branching point determination that considers distance to center, target segment length, branching factor, target token logprob distribution, etc.
    TempoSpontaneous, // we make the model to always rollout from scratch, and derive the branching point from the divergence point
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy, clap::ValueEnum)]
pub enum AdvantageCalculationPolicy {
    TreeMappoPosterior, // our method that make assumptions about the relationship between each segment's contribution and final outcome, and then use probabilistic model and maximum-a-posteriori update to get the contribution posteriors
    TreeRpoWinRate, // the advantage is linearly proportional to num_wins / total_plays of a segment or node's children outcomes
}

fn default_early_stopping_decision_trajectories() -> usize {
    8
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DirectRolloutConfig<S: DatasetSplit> {
    // pub split: DatasetSplit,
    pub branching_policy: BranchingPolicy,
    pub max_num_trunks: usize,
    #[serde(default = "default_early_stopping_decision_trajectories")]
    pub early_stopping_decision_trajectories: usize,
    pub max_num_total_trajectories: usize,
    pub fixed_temperature: NotNan<f32>,
    /// When set, the training question rotation uses this fixed seed for every
    /// epoch instead of the epoch number, so all epochs cover the same
    /// question segment (used by positive-control experiments).
    #[serde(default)]
    pub question_rotation_seed: Option<u64>,
    pub use_tool: bool,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<S>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_tool::hybrid_dataset::{Training, Validation};

    #[test]
    fn rollout_config_without_rotation_seed_defaults_to_none() {
        let json = r#"{
            "branching_policy": "TreeMappoGuided",
            "max_num_trunks": 4,
            "early_stopping_decision_trajectories": 6,
            "max_num_total_trajectories": 16,
            "fixed_temperature": 0.7,
            "use_tool": false
        }"#;
        let config: DirectRolloutConfig<Training> = serde_json::from_str(json).unwrap();
        assert_eq!(config.question_rotation_seed, None);
    }

    #[test]
    fn rollout_config_with_rotation_seed_deserializes() {
        let json = r#"{
            "branching_policy": "TreeMappoGuided",
            "max_num_trunks": 4,
            "early_stopping_decision_trajectories": 6,
            "max_num_total_trajectories": 16,
            "fixed_temperature": 0.7,
            "question_rotation_seed": 0,
            "use_tool": false
        }"#;
        let config: DirectRolloutConfig<Training> = serde_json::from_str(json).unwrap();
        assert_eq!(config.question_rotation_seed, Some(0));
    }

    #[test]
    fn shipped_control_and_t0_configs_parse() {
        let control = std::fs::read_to_string("config/rollout_config_training_notool_control.json")
            .expect("control rollout config must exist");
        let control_config: DirectRolloutConfig<Training> =
            serde_json::from_str(&control).unwrap();
        assert_eq!(control_config.question_rotation_seed, Some(0));

        let validation_t0 =
            std::fs::read_to_string("config/rollout_config_validation_notool_t0.json")
                .expect("t0 validation rollout config must exist");
        let validation_config: DirectRolloutConfig<Validation> =
            serde_json::from_str(&validation_t0).unwrap();
        assert_eq!(validation_config.question_rotation_seed, None);
        assert_eq!(validation_config.fixed_temperature.into_inner(), 0.0);
    }
}
