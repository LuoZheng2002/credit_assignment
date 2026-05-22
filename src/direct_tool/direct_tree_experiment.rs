use serde::{Deserialize, Serialize};

// GRPO is a special case where there are only trunks and the credit assignment is like MonteCarloTree
// then the advantages are normalized

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectTreeExperiment {
    TemperatureToAccuracy {
        temperature: f32, // the max number of trunks is 1 and the max total trajectories is 1
    },
    Grpo {
        // initial temperature is 1.0
        // the max number of trajectories is equal to the max number of trunks, so that there will be no branches
        // uses win/loss ratio to determine the step advantage, normalized within the tree
        max_num_trunks: usize, // equals to the max number of trajectories
    },
    TreeMappo {
        // our algorithm, uses the EM fitting algorithm to calculate the advantage
        max_num_trunks: usize,             // should be 4
        max_num_total_trajectories: usize, // can vary with ablation setting
    },
    NaturalDivergence {
        // The TEMPO paper's method, may be tricky to implement
        max_num_total_trajectories: usize,
    },
}
