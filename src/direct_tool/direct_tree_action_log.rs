use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file}, progress_tui_server::log_warning, sqlite_store::{SqliteBusyRetryConfig, SqliteStore}, sqlite_table_array_store::SqliteTableArrayStore
};
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, sync::Arc};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree_action::DirectTreeAction,
        hybrid_dataset::{AssetFileHybridDataset, HybridDatasetQuestion},
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::{read_json, write_json},
    llm_model::LlmModelMarker,
};

const ACTION_LOG_SCHEMA_VERSION: usize = 3;

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct DirectTreeActionLog<M> {
    pub question: HybridDatasetQuestion,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub actions: Vec<DirectTreeAction<M>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionLogMetadata {
    pub question: HybridDatasetQuestion,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
}

#[derive(Debug)]
pub struct DirectTreeActionLogStore<M: LlmModelMarker> {
    metadata_store: Arc<SqliteStore<usize, DirectTreeActionLogMetadata>>,
    action_store: Arc<SqliteTableArrayStore<usize, DirectTreeAction<M>>>,
    _phantom: PhantomData<M>,
}

impl<M: LlmModelMarker> Clone for DirectTreeActionLogStore<M> {
    fn clone(&self) -> Self {
        Self {
            metadata_store: Arc::clone(&self.metadata_store),
            action_store: Arc::clone(&self.action_store),
            _phantom: PhantomData,
        }
    }
}

impl<M: LlmModelMarker> DirectTreeActionLogStore<M> {
    pub async fn initialize_if_missing(db_path: impl Into<String>) -> Self {
        let db_path = db_path.into();
        let metadata_store =
            SqliteStore::<usize, DirectTreeActionLogMetadata>::initialize_if_missing(
                db_path.clone(),
            )
            .await;
        let action_store = SqliteTableArrayStore::<usize, DirectTreeAction<M>>::new(db_path)
            .await
            .expect("failed to initialize sqlite table array store for direct action log actions");
        Self {
            metadata_store: Arc::new(metadata_store),
            action_store: Arc::new(action_store),
            _phantom: PhantomData,
        }
    }

    pub async fn get_keys(&self) -> Result<Vec<usize>, String> {
        self.metadata_store.get_keys().await
    }

    pub async fn get(&self, key: usize) -> Result<Option<DirectTreeActionLog<M>>, String> {
        let metadata = self.metadata_store.get(key).await?;
        let Some(metadata) = metadata else {
            return Ok(None);
        };
        let indexed_actions = self.action_store.load_table_with_indices(key).await?;
        for (expected_index, (stored_index, _)) in indexed_actions.iter().enumerate() {
            if *stored_index != expected_index {
                return Err(format!(
                    "Non-contiguous action indices for key {}: expected {}, got {}",
                    key, expected_index, stored_index
                ));
            }
        }
        let actions = indexed_actions
            .into_iter()
            .map(|(_, action)| action)
            .collect();
        Ok(Some(DirectTreeActionLog {
            question: metadata.question,
            rollout_config: metadata.rollout_config,
            posterior_calculation_config: metadata.posterior_calculation_config,
            actions,
        }))
    }

    pub async fn get_or_init_metadata(
        &self,
        key: usize,
        default_metadata: &DirectTreeActionLogMetadata,
    ) -> Result<DirectTreeActionLogMetadata, String> {
        let existing = self.metadata_store.get(key).await?;
        if let Some(existing) = existing {
            return Ok(existing);
        }
        self.metadata_store
            .upsert(key, default_metadata, SqliteBusyRetryConfig::aggressive())
            .await?;
        Ok(default_metadata.clone())
    }

    pub async fn append_action_at(
        &self,
        key: usize,
        action_index: usize,
        action: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        self.action_store.append_at(key, action_index, action).await
    }
}

impl<M> Clone for DirectTreeActionLog<M> {
    fn clone(&self) -> Self {
        Self {
            question: self.question.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            actions: self.actions.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutPurpose {
    Training,
    Evaluation,
    Testing {},
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileDirectTreeActionLogsTracking {
    pub dataset_hash: Base64Hash,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub action_log_schema_version: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssetFileDirectTreeActionLogs<M: LlmModelMarker> {
    pub nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,

    pub epoch: usize, // the epoch index
    #[serde(skip)]
    pub _phantom: PhantomData<M>,
}

// for this asset, we check if the tracking file is stale
// if so, we delete the target file

impl<M: LlmModelMarker> AssetFileDirectTreeActionLogs<M> {
    fn to_short_hash(&self) -> String {
        let serialized = serde_json::to_vec(self).unwrap();
        let hash = blake3::hash(&serialized);
        let short_hash = hex::encode(&hash.as_bytes()[..4]); // Take the first 4 bytes for a shorter hash
        assert_eq!(short_hash.len(), 8); // 4 bytes should give us 8 hex characters
        short_hash
    }
    pub fn file_path(&self) -> String {
        format!(
            "results/{}/{}/epoch_{}/action_logs_{}.sqlite",
            M::CLI_NAME,
            self.nickname,
            self.epoch,
            self.to_short_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/{}/epoch_{}/action_logs_{}_tracking.json",
            M::CLI_NAME,
            self.nickname,
            self.epoch,
            self.to_short_hash()
        )
    }
    pub fn delete_target_file_if_stale(&self) {
        let stale_reason: Option<String> = match read_json::<AssetFileDirectTreeActionLogsTracking>(
            self.version_tracking_path(),
        ) {
            Ok(tracking_content) => {
                let dataset_asset_file = AssetFileHybridDataset {
                    split: self.rollout_config.split.clone(),
                };
                let dataset_hash = futures::executor::block_on(dataset_asset_file.synchronize());
                if dataset_hash != tracking_content.dataset_hash {
                    Some(format!(
                        "Tracking file exists but stale: dataset hash mismatch (tracking: {:?}, current: {:?})",
                        tracking_content.dataset_hash, dataset_hash
                    ))
                } else if tracking_content.action_log_schema_version != ACTION_LOG_SCHEMA_VERSION {
                    Some(format!(
                        "Tracking file exists but stale: action log schema version mismatch (tracking: {}, current: {})",
                        tracking_content.action_log_schema_version, ACTION_LOG_SCHEMA_VERSION
                    ))
                } else if tracking_content.rollout_config != self.rollout_config {
                    Some(format!(
                        "Tracking file exists but stale: rollout config mismatch (tracking: {:?}, current: {:?})",
                        tracking_content.rollout_config, self.rollout_config
                    ))
                } else if tracking_content.posterior_calculation_config
                    != self.posterior_calculation_config
                {
                    Some(format!(
                        "Tracking file exists but stale: posterior calculation config mismatch (tracking: {:?}, current: {:?})",
                        tracking_content.posterior_calculation_config,
                        self.posterior_calculation_config
                    ))
                } else if tracking_content.config_nickname != self.nickname {
                    Some(format!(
                        "Tracking file exists but stale: nickname mismatch (tracking: {:?}, current: {:?})",
                        tracking_content.config_nickname, self.nickname
                    ))
                } else {
                    None
                }
            }
            Err(_) => {
                // if we cannot read the tracking file, we consider the target file as stale (if exists)
                Some("Tracking file missing or incompatible format".to_string())
            }
        };
        if let Some(reason) = stale_reason {
            log_warning(&format!(
                "Target file {} is stale. Deleting if exists... Reason: {}",
                self.file_path(),
                reason
            ));
            // the target file is stale, delete it if exists
            if std::path::Path::new(&self.file_path()).exists() {
                std::fs::remove_file(self.file_path()).expect("Failed to delete stale target file");
                log_warning("Deleted stale target file for direct action log");
            }
        }
    }
    pub fn create_tracking_file(&self) {
        // we collect the dataset hash
        let dataset_asset_file = AssetFileHybridDataset {
            split: self.rollout_config.split.clone(),
        };
        let dataset_hash = futures::executor::block_on(dataset_asset_file.synchronize());
        let tracking_content = AssetFileDirectTreeActionLogsTracking {
            dataset_hash,
            config_nickname: self.nickname.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            action_log_schema_version: ACTION_LOG_SCHEMA_VERSION,
            epoch: self.epoch,
        };
        write_json(self.version_tracking_path(), &tracking_content).unwrap();
    }
}

#[async_trait::async_trait]
impl<M: LlmModelMarker> AssetFile for AssetFileDirectTreeActionLogs<M> {
    type FileModel = DirectTreeActionLogStore<M>;
    async fn synchronize(&self) -> Base64Hash {
        // synchromize all dependency assets
        let dataset_asset_file = AssetFileHybridDataset {
            split: self.rollout_config.split.clone(),
        };
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
        assert_eq!(
            tracking_content.posterior_calculation_config,
            self.posterior_calculation_config
        );
        assert_eq!(tracking_content.config_nickname, self.nickname);
        // check if target file exists and returns hash
        hash_file(self.file_path()).expect("Target file missing for direct action log")
    }
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        DirectTreeActionLogStore::<M>::initialize_if_missing(self.file_path()).await
    }
}
