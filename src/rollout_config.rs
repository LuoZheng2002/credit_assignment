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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DirectRolloutConfig<S: DatasetSplit> {
    // pub split: DatasetSplit,
    pub branching_policy: BranchingPolicy,
    pub num_trunks: usize,
    pub num_early_stopping_leaves: usize,
    pub num_leaves: usize,
    pub fixed_temperature: NotNan<f32>,
    pub use_tool: bool,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<S>,
}
