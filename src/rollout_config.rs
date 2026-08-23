use serde::{Deserialize, Serialize};

use crate::hybrid_dataset::Training;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, clap::ValueEnum)]
pub enum BranchingPolicy {
    TreeMappoGuided, // our method with guided branching point determination that considers distance to center, target segment length, branching factor, target token logprob distribution, etc.
    TempoSpontaneous, // we make the model to always rollout from scratch, and derive the branching point from the divergence point
    TreeRlEntropyGuided, // TreeRL-style ablation: choose guided branch points by token-distribution entropy while keeping the same tree infrastructure
}

impl BranchingPolicy {
    pub fn abbreviation(&self) -> &'static str {
        match self {
            BranchingPolicy::TreeMappoGuided => "TMB",
            BranchingPolicy::TempoSpontaneous => "TPB",
            BranchingPolicy::TreeRlEntropyGuided => "TRLEB",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            BranchingPolicy::TreeMappoGuided => "TreeMAPPO guided branching",
            BranchingPolicy::TempoSpontaneous => "TEMPO-style prefix-tree branching",
            BranchingPolicy::TreeRlEntropyGuided => "TreeRL-style entropy-guided branching",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy, clap::ValueEnum)]
pub enum TrainingAdvantagePolicy {
    TreeMappoPosterior, // our method that make assumptions about the relationship between each segment's contribution and final outcome, and then use probabilistic model and maximum-a-posteriori update to get the contribution posteriors
    TreeRpoWinRate, // TreeRPO-style child-group win-rate advantage with bottom-up outcome backup
    TreeRlLocalGlobal, // TreeRL-style ablation: combine local parent-child and global root-child value deltas
    GrpoTerminalReward, // flat GRPO baseline: group-normalized terminal correctness assigned to each response's supervised tokens
}

impl TrainingAdvantagePolicy {
    pub fn abbreviation(&self) -> &'static str {
        match self {
            TrainingAdvantagePolicy::TreeMappoPosterior => "TMA",
            TrainingAdvantagePolicy::TreeRpoWinRate => "TRPOA",
            TrainingAdvantagePolicy::TreeRlLocalGlobal => "TRLA",
            TrainingAdvantagePolicy::GrpoTerminalReward => "GRPOA",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            TrainingAdvantagePolicy::TreeMappoPosterior => "TreeMAPPO posterior segment advantage",
            TrainingAdvantagePolicy::TreeRpoWinRate => {
                "TreeRPO-style child-group outcome advantage"
            }
            TrainingAdvantagePolicy::TreeRlLocalGlobal => {
                "TreeRL-style local-global value advantage"
            }
            TrainingAdvantagePolicy::GrpoTerminalReward => {
                "GRPO-style group-normalized terminal reward advantage"
            }
        }
    }
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
