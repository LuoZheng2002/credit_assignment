use research_utility::{asset_file::Base64Hash, sqlite_store::SqliteStore};
use serde::{Deserialize, Serialize};

use crate::{direct_tool::{direct_rollout_config::DirectRolloutConfig, direct_tree::DirectTree, direct_tree_action_log::DirectTreeActionLog, hybrid_dataset::HybridDatasetQuestion, posterior_calculation_config::PosteriorCalculationConfig}, llm_model::LlmModelName};

pub async fn rollout_logs_to_training_trajectories(
    action_log_store: SqliteStore<usize, DirectTreeActionLog>,
) -> Vec<DirectTrainingTrajectory> {
    // iterate through all action logs
    let mut keys = action_log_store.get_keys().await.unwrap();
    keys.sort(); // ensure deterministic order
    for key in keys {
        let action_log = action_log_store.get(key).await.unwrap().unwrap();
        // convert each action log into a training trajectory
    }

    todo!()
}

fn action_log_to_candidate_trajectories(
    action_log: DirectTreeActionLog,
) -> Vec<DirectTrainingTrajectory> {
    // todo: resume after making DirectTree to be non-generic

    todo!()
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectTrainingTrajectory {
    pub question: HybridDatasetQuestion,
    pub input_ids: Vec<i32>,
    pub labels: Vec<i32>, // we may not need to let model learn to stop after tool_wait or end since our framework already handled this
    pub advantages: Vec<f32>,
    pub average_segment_advantage: f32,
}

pub struct AssetFileTrainingTrajectories {
    pub model: LlmModelName,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
}

pub struct AssetFileTrainingTrajectoriesTracking {
    pub rollout_log_hash: Base64Hash,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub tokenized_schema_version: usize,
}


