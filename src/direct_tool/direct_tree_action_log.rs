use research_utility::asset_file::Base64Hash;
use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{direct_tree_action::DirectTreeAction, hybrid_dataset::HybridDatasetQuestion},
    llm_model::LlmModelName,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionLog {
    pub question: HybridDatasetQuestion,
    pub actions: Vec<DirectTreeAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileActionLogsTracking {
    pub dataset_hash: Base64Hash,
    pub action_log_schema_version: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssetFileDirectTreeActionLogs {
    model: LlmModelName,
}
