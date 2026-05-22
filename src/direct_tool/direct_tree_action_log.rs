use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    sqlite_store::SqliteStore,
};
use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        direct_tree_action::DirectTreeAction,
        hybrid_dataset::{AssetFileHybridDataset, HybridDatasetQuestion},
    },
    json_line_util::read_json,
    llm_model::LlmModelName,
};

const ACTION_LOG_SCHEMA_VERSION: usize = 1;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionLog {
    pub question: HybridDatasetQuestion,
    pub rollout_config: DirectRolloutConfig,
    pub actions: Vec<DirectTreeAction>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileDirectTreeActionLogsTracking {
    pub dataset_hash: Base64Hash,
    pub action_log_schema_version: usize,
    pub config: DirectRolloutConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssetFileDirectTreeActionLogs {
    pub model: LlmModelName,
    pub config: DirectRolloutConfig,
}

impl AssetFileDirectTreeActionLogs {
    fn file_path(&self) -> String {
        format!(
            "results/{}/direct_action_Log_{}.sqlite",
            self.model.cli_name(),
            self.config.to_short_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/direct_action_Log_{}_tracking.json",
            self.model.cli_name(),
            self.config.to_short_hash()
        )
    }
}

#[async_trait::async_trait]
impl AssetFile for AssetFileDirectTreeActionLogs {
    type FileModel = SqliteStore<usize, DirectTreeActionLog>;
    async fn synchronize(&self) -> Base64Hash {
        // synchromize all dependency assets
        let dataset_asset_file = AssetFileHybridDataset;
        let dataset_hash = dataset_asset_file.synchronize().await;
        let tracking_content =
            read_json::<AssetFileDirectTreeActionLogsTracking>(self.version_tracking_path())
                .expect("Tracking file missing for direct action log");

        assert_eq!(dataset_hash, tracking_content.dataset_hash);
        assert_eq!(
            tracking_content.action_log_schema_version,
            ACTION_LOG_SCHEMA_VERSION
        );
        assert_eq!(tracking_content.config, self.config);
        // check if target file exists and returns hash
        hash_file(self.file_path()).expect("Target file missing for direct action log")
    }
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::assume_initialized(self.file_path()).await
    }
}
