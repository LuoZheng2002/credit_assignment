use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    sqlite_store::SqliteStore,
};
use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree_action::DirectTreeAction,
        hybrid_dataset::{AssetFileHybridDataset, HybridDatasetQuestion},
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::read_json,
    llm_model::LlmModelName,
};

const ACTION_LOG_SCHEMA_VERSION: usize = 1;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionLog {
    pub question: HybridDatasetQuestion,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub actions: Vec<DirectTreeAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileDirectTreeActionLogsTracking {
    pub dataset_hash: Base64Hash,
    pub action_log_schema_version: usize,
    pub rollout_config: DirectRolloutConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssetFileDirectTreeActionLogs {
    pub model: LlmModelName,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
}

// for this asset, we check if the tracking file is stale
// if so, we delete the target file

impl AssetFileDirectTreeActionLogs {
    pub fn file_path(&self) -> String {
        format!(
            "results/{}/direct_action_Log_{}.sqlite",
            self.model.cli_name(),
            self.rollout_config.to_short_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/direct_action_Log_{}_tracking.json",
            self.model.cli_name(),
            self.rollout_config.to_short_hash()
        )
    }
    pub fn delete_target_file_if_stale(&self) {
        let stale = match read_json::<AssetFileDirectTreeActionLogsTracking>(
            self.version_tracking_path(),
        ) {
            Ok(tracking_content) => {
                let dataset_asset_file = AssetFileHybridDataset;
                let dataset_hash = futures::executor::block_on(dataset_asset_file.synchronize());
                dataset_hash != tracking_content.dataset_hash
                    || tracking_content.action_log_schema_version != ACTION_LOG_SCHEMA_VERSION
                    || tracking_content.rollout_config != self.rollout_config
            }
            Err(_) => {
                // if we cannot read the tracking file, we consider the target file as stale (if exists)
                true
            }
        };
        if stale {
            println!(
                "Target file {} is stale. Deleting if exists...",
                self.file_path()
            );
            // the target file is stale, delete it if exists
            if std::path::Path::new(&self.file_path()).exists() {
                std::fs::remove_file(self.file_path()).expect("Failed to delete stale target file");
                println!("Deleted stale target file for direct action log");
            }
        }
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
        assert_eq!(tracking_content.rollout_config, self.rollout_config);
        // check if target file exists and returns hash
        hash_file(self.file_path()).expect("Target file missing for direct action log")
    }
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::assume_initialized(self.file_path()).await
    }
}
