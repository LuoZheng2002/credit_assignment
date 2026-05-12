use serde::{Deserialize, Serialize};

use crate::{
    agent::tree_schema::CompletedTree,
    direct_answer::generate_raw_answers::LlmModel,
    em::em_schema::EmFitPerTree,
    version_tracking::{AssetFile, Base64Hash},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdvantageCompositionPerTree {
    pub question_id: usize,
    pub per_node: Vec<AdvantageCompositionPerNode>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdvantageCompositionPerNode {
    pub node_id: usize,
    pub contribution_mean: f64, // when displayed, normalize only the scaling factor, and set pure red / green color beyond 95% ci
    pub contribution_log_std: f64, // when displayed, normalize both the mean and std, and set pure red / green color beyond 95% ci
    pub contribution_mean_div_var: f64, // when displayed, normalize only the scaling factor, and set pure red / green color beyond 95% ci
    pub contribution_mean_div_var_normalized: f64, // this already normalizes both the mean and std within the tree. For display, it should be multiplied by a weight factor, and then set pure red / green color beyond 95% ci of N(0, 1)
    pub step_quality_tool_advantage_normalized: f64, // normalized across all trees to N(0, 1). For display, multiplied by a weight factor, and then set pure red / green color beyond 95% ci of N(0, 1)
    pub step_quality_complete_advantage_normalized: f64, // same as above
    pub step_quality_focused_advantage_normalized: f64, // same as above
    pub trajectory_advantage_normalized: f64,        // same as above
}

pub struct AssetFileAdvantageComposition {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
}
impl AssetFileAdvantageComposition {
    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_advantage_composition_{}.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }
    pub fn compose_advantage(
        trees: &[CompletedTree],
        em_fit_per_tree: &[EmFitPerTree],
    ) -> Vec<AdvantageCompositionPerTree> {
        todo!()
    }
}

impl AssetFile for AssetFileAdvantageComposition {
    type FileModel = Vec<AdvantageCompositionPerTree>;

    fn synchronize(&self) -> Base64Hash {
        todo!()
    }

    fn fetch(&self) -> Self::FileModel {
        todo!()
    }
}
