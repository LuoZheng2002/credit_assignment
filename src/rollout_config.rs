use serde::{Deserialize, Serialize};

use crate::hybrid_dataset::Training;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum BranchingPolicy {
    TreeMappoGuided, // our method with guided branching point determination that considers distance to center, target segment length, branching factor, target token logprob distribution, etc.
    TempoSpontaneous, // we make the model to always rollout from scratch, and derive the branching point from the divergence point
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy, clap::ValueEnum)]
pub enum TrainingAdvantagePolicy {
    TreeMappoPosterior, // our method that make assumptions about the relationship between each segment's contribution and final outcome, and then use probabilistic model and maximum-a-posteriori update to get the contribution posteriors
    TreeRpoWinRate, // the advantage is linearly proportional to num_wins / total_plays of a segment or node's children outcomes
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RolloutConfig<S> {
    pub branching_policy: BranchingPolicy,
    pub num_trunks: usize,
    pub num_early_stopping_leaves: usize,
    pub num_leaves: usize,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<S>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TrainingRolloutConfig {
    pub branching_policy: BranchingPolicy,
    pub num_trunks: usize,
    pub num_early_stopping_leaves: usize,
    pub num_leaves: usize,
    pub training_advantage_policy: TrainingAdvantagePolicy,
}

impl TrainingRolloutConfig {
    pub fn to_rollout_config(&self) -> RolloutConfig<Training> {
        RolloutConfig {
            branching_policy: self.branching_policy.clone(),
            num_trunks: self.num_trunks,
            num_early_stopping_leaves: self.num_early_stopping_leaves,
            num_leaves: self.num_leaves,
            _phantom: std::marker::PhantomData,
        }
    }
}
