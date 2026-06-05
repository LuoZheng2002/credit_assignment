use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    progress_tui_server::log_warning,
    sqlite_table_array_store::SqliteTableArrayStore,
};
use minijinja::context;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, sync::LazyLock};
use tokio::sync::{mpsc, oneshot};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree_action::DirectTreeAction,
        hybrid_dataset::{
            AssetFileHybridDataset, DatasetSplit, HybridDatasetQuestion, QuestionFlatId,
        },
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::{read_json, write_json},
    llm_model::LlmModelMarker,
};

const ACTION_LOG_SCHEMA_VERSION: usize = 3;
const ACTION_LOGS_PARENT_DIR_TEMPLATE_PATH: &str = "config/training/action_logs_parent_dir.jinja";

fn load_template_environment(
    template_path: &str,
    template_name: &'static str,
) -> Result<minijinja::Environment<'static>, String> {
    let template_source = std::fs::read_to_string(template_path)
        .map_err(|err| format!("Failed to read {}: {}", template_path, err))?;
    let mut env = minijinja::Environment::new();
    env.add_template_owned(template_name, template_source)
        .map_err(|err| format!("Failed to parse {} template: {}", template_name, err))?;
    Ok(env)
}

fn action_logs_parent_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    static ACTION_LOGS_PARENT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
        Result<minijinja::Environment<'static>, String>,
    > = LazyLock::new(|| {
        load_template_environment(ACTION_LOGS_PARENT_DIR_TEMPLATE_PATH, "action_logs_parent_dir")
    });

    let env = ACTION_LOGS_PARENT_DIR_TEMPLATE_ENVIRONMENT
        .as_ref()
        .map_err(|err| err.clone())?;
    let template = env
        .get_template("action_logs_parent_dir")
        .map_err(|err| format!("Failed to load action_logs_parent_dir template: {}", err))?;
    let rendered = template
        .render(context! {
            model_cli_name => model_cli_name,
            config_nickname => config_nickname,
            epoch => epoch,
        })
        .map_err(|err| format!("Failed to render action_logs_parent_dir template: {}", err))?;
    let rendered = rendered.trim().to_string();
    if rendered.is_empty() {
        return Err("Rendered action_logs_parent_dir template is empty".to_string());
    }
    Ok(rendered)
}

#[derive(Clone)]
pub struct DirectTreeActionLog<M: LlmModelMarker, S: DatasetSplit> {
    pub question: HybridDatasetQuestion<S>,
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub actions: Vec<DirectTreeAction<M>>,
}

// #[derive(Serialize, Deserialize, Debug, Clone)]
// pub struct DirectTreeActionLogMetadata {
//     pub question: HybridDatasetQuestion,
//     pub rollout_config: DirectRolloutConfig,
//     pub posterior_calculation_config: PosteriorCalculationConfig,
// }

#[derive(Debug)]
pub struct DirectTreeActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    // pub metadata_store: SqliteStore<usize, DirectTreeActionLogMetadata>,
    pub action_store: SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>>,
    pub _phantom: PhantomData<M>,
}

#[derive(Debug, Clone)]
pub struct ActionStoreAdapter<M: LlmModelMarker, S: DatasetSplit> {
    request_tx: mpsc::UnboundedSender<StoreRequest<M, S>>,
    _phantom: PhantomData<(M, S)>,
}

enum StoreRequest<M: LlmModelMarker, S: DatasetSplit> {
    // GetKeys {
    //     response_tx: oneshot::Sender<Result<Vec<usize>, String>>,
    // },
    // Get {
    //     key: usize,
    //     response_tx: oneshot::Sender<Result<Option<DirectTreeActionLog<M>>, String>>,
    // },
    // GetOrInitMetadata {
    //     key: usize,
    //     default_metadata: DirectTreeActionLogMetadata,
    //     response_tx: oneshot::Sender<Result<DirectTreeActionLogMetadata, String>>,
    // },
    GetOrInitActions {
        key: QuestionFlatId<S>,
        response_tx: oneshot::Sender<Result<Vec<DirectTreeAction<M>>, String>>,
    },
    AppendActionAt {
        key: QuestionFlatId<S>,
        action_index: usize,
        action: DirectTreeAction<M>,
        response_tx: oneshot::Sender<Result<(), String>>,
    },
}

// impl<M: LlmModelMarker> Clone for ActionStoreAdapter<M> {
//     fn clone(&self) -> Self {
//         Self {
//             request_tx: self.request_tx.clone(),
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<M: LlmModelMarker> DirectTreeActionLogStore<M> {
//     pub fn initialize_if_missing(db_path: impl Into<String>) -> Self {
//         let db_path = db_path.into();
//         // let metadata_store =
//         //     SqliteStore::<usize, DirectTreeActionLogMetadata>::initialize_if_missing(
//         //         db_path.clone(),
//         //     );
//         // let action_store = SqliteTableArrayStore::<usize, DirectTreeAction<M>>::new(db_path)
//         //     .expect("failed to initialize sqlite table array store for direct action log actions");
//         let action_store =
//             SqliteTableArrayStore::<usize, DirectTreeAction<M>>::initialize_if_missing(db_path);
//         Self {
//             // metadata_store,
//             action_store,
//             _phantom: PhantomData,
//         }
//     }
//     pub fn get(&self, key: usize) -> Result<Option<DirectTreeActionLog<M>>, String> {
//         // let metadata = self.metadata_store.get(key)?;
//         // if let Some(metadata) = metadata {
//         //     let actions = self.action_store.load_table_sorted(key)?;
//         //     Ok(Some(DirectTreeActionLog::from_metadata_and_actions(
//         //         metadata, actions,
//         //     )))
//         // } else {
//         //     Ok(None)
//         // }
//         let actions = self.action_store.load_table_sorted(key)?;
//         DirectTreeActionLog{

//         }
//     }
//     pub fn get_keys(&self) -> Result<Vec<usize>, String> {
//         self.metadata_store.get_keys()
//     }

//     // pub fn get_or_init_metadata(
//     //     &self,
//     //     key: usize,
//     //     default_metadata: DirectTreeActionLogMetadata,
//     // ) -> Result<DirectTreeActionLogMetadata, String> {
//     //     let existing = self.metadata_store.get(key)?;
//     //     if let Some(existing) = existing {
//     //         return Ok(existing);
//     //     }
//     //     self.metadata_store
//     //         .upsert(key, default_metadata, SqliteBusyRetryConfig::aggressive())?;
//     //     Ok(default_metadata)
//     // }

//     pub fn append_action_at(
//         &self,
//         key: usize,
//         action_index: usize,
//         action: &DirectTreeAction<M>,
//     ) -> Result<(), String> {
//         self.action_store.append_at(key, action_index, action)
//     }
// }

impl<M: LlmModelMarker, S: DatasetSplit> ActionStoreAdapter<M, S> {
    pub fn new(store: SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>>) -> Self {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        Self::spawn_worker(store, request_rx);
        Self {
            request_tx,
            _phantom: PhantomData,
        }
    }

    fn spawn_worker(
        store: SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>>,
        mut request_rx: mpsc::UnboundedReceiver<StoreRequest<M, S>>,
    ) {
        std::thread::Builder::new()
            .name("direct_action_log_sqlite_worker".to_string())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to initialize direct action log sqlite worker runtime");
                runtime.block_on(async move {
                    while let Some(request) = request_rx.recv().await {
                        match request {
                            // StoreRequest::GetKeys { response_tx } => {
                            //     let result = store.get_keys();
                            //     let _ = response_tx.send(result);
                            // }
                            // StoreRequest::Get { key, response_tx } => {
                            //     let result = store.get(key);
                            //     let _ = response_tx.send(result);
                            // }
                            // StoreRequest::GetOrInitMetadata {
                            //     key,
                            //     default_metadata,
                            //     response_tx,
                            // } => {
                            //     let result = store.get_or_init_metadata(key, &default_metadata);
                            //     let _ = response_tx.send(result);
                            // }
                            StoreRequest::GetOrInitActions { key, response_tx } => {
                                let result = store.load_or_init_table_sorted(key, || Vec::new());
                                let _ = response_tx.send(result);
                            }
                            StoreRequest::AppendActionAt {
                                key,
                                action_index,
                                action,
                                response_tx,
                            } => {
                                let result = store.append_at(key, action_index, &action);
                                let _ = response_tx.send(result);
                            }
                        }
                    }
                });
            })
            .expect("failed to spawn direct action log sqlite worker thread");
    }

    // pub async fn get_keys(&self) -> Result<Vec<usize>, String> {
    //     let (response_tx, response_rx) = oneshot::channel();
    //     self.request_tx
    //         .send(StoreRequest::GetKeys { response_tx })
    //         .map_err(|_| "direct action log sqlite worker has shut down".to_string())?;
    //     response_rx
    //         .await
    //         .map_err(|_| "direct action log sqlite worker response dropped".to_string())?
    // }

    // pub async fn get(&self, key: usize) -> Result<Option<DirectTreeActionLog<M>>, String> {
    //     let (response_tx, response_rx) = oneshot::channel();
    //     self.request_tx
    //         .send(StoreRequest::Get { key, response_tx })
    //         .map_err(|_| "direct action log sqlite worker has shut down".to_string())?;
    //     response_rx
    //         .await
    //         .map_err(|_| "direct action log sqlite worker response dropped".to_string())?
    // }

    // pub async fn get_or_init_metadata(
    //     &self,
    //     key: usize,
    //     default_metadata: &DirectTreeActionLogMetadata,
    // ) -> Result<DirectTreeActionLogMetadata, String> {
    //     let (response_tx, response_rx) = oneshot::channel();
    //     self.request_tx
    //         .send(StoreRequest::GetOrInitMetadata {
    //             key,
    //             default_metadata: default_metadata.clone(),
    //             response_tx,
    //         })
    //         .map_err(|_| "direct action log sqlite worker has shut down".to_string())?;
    //     response_rx
    //         .await
    //         .map_err(|_| "direct action log sqlite worker response dropped".to_string())?
    // }

    pub async fn get_or_init_actions(
        &self,
        key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(StoreRequest::GetOrInitActions { key, response_tx })
            .map_err(|_| "direct action log sqlite worker has shut down".to_string())?;
        response_rx
            .await
            .map_err(|_| "direct action log sqlite worker response dropped".to_string())?
    }

    pub async fn append_action_at(
        &self,
        key: QuestionFlatId<S>,
        action_index: usize,
        action: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(StoreRequest::AppendActionAt {
                key,
                action_index,
                action: action.clone(),
                response_tx,
            })
            .map_err(|_| "direct action log sqlite worker has shut down".to_string())?;
        response_rx
            .await
            .map_err(|_| "direct action log sqlite worker response dropped".to_string())?
    }
}

// impl<M> DirectTreeActionLog<M> {
//     pub fn from_metadata_and_actions(
//         metadata: DirectTreeActionLogMetadata,
//         actions: Vec<DirectTreeAction<M>>,
//     ) -> Self {
//         Self {
//             question: metadata.question,
//             rollout_config: metadata.rollout_config,
//             posterior_calculation_config: metadata.posterior_calculation_config,
//             actions,
//         }
//     }
// }

// impl<M,> Clone for DirectTreeActionLog<M> {
//     fn clone(&self) -> Self {
//         Self {
//             question: self.question.clone(),
//             rollout_config: self.rollout_config.clone(),
//             posterior_calculation_config: self.posterior_calculation_config.clone(),
//             actions: self.actions.clone(),
//         }
//     }
// }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutPurpose {
    Training,
    Evaluation,
    Testing {},
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "S: DatasetSplit"))]
pub struct AssetFileDirectTreeActionLogsTracking<S: DatasetSplit> {
    pub dataset_hash: Base64Hash,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub action_log_schema_version: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound(deserialize = "S: DatasetSplit"))]
pub struct AssetFileDirectTreeActionLogs<M: LlmModelMarker, S: DatasetSplit> {
    pub nickname: String,
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    #[serde(skip)]
    pub _phantom: PhantomData<M>,
}

// for this asset, we check if the tracking file is stale
// if so, we delete the target file

impl<M: LlmModelMarker, S: DatasetSplit> AssetFileDirectTreeActionLogs<M, S> {
    fn to_short_hash(&self) -> String {
        let serialized = serde_json::to_vec(self).unwrap();
        let hash = blake3::hash(&serialized);
        let short_hash = hex::encode(&hash.as_bytes()[..4]); // Take the first 4 bytes for a shorter hash
        assert_eq!(short_hash.len(), 8); // 4 bytes should give us 8 hex characters
        short_hash
    }
    pub fn actions_file_path(&self) -> String {
        let parent_dir = action_logs_parent_dir_from_template(M::CLI_NAME, &self.nickname, self.epoch)
            .unwrap_or_else(|err| panic!("Failed to resolve action logs parent directory: {}", err));
        format!("{}/action_logs_{}.sqlite", parent_dir, self.to_short_hash())
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
        let stale_reason: Option<String> = match read_json::<AssetFileDirectTreeActionLogsTracking<S>>(
            self.version_tracking_path(),
        ) {
            Ok(tracking_content) => {
                let dataset_asset_file = AssetFileHybridDataset::<S>(PhantomData);
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
                self.actions_file_path(),
                reason
            ));
            // the target file is stale, delete it if exists
            if std::path::Path::new(&self.actions_file_path()).exists() {
                std::fs::remove_file(self.actions_file_path())
                    .expect("Failed to delete stale target file");
                log_warning("Deleted stale target file for direct action log");
            }
        }
    }
    pub fn create_tracking_file(&self) {
        // we collect the dataset hash
        let dataset_asset_file = AssetFileHybridDataset::<S>(PhantomData);
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
impl<M: LlmModelMarker, S: DatasetSplit> AssetFile for AssetFileDirectTreeActionLogs<M, S> {
    type FileModel = SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>>;
    async fn synchronize(&self) -> Base64Hash {
        // synchromize all dependency assets
        let dataset_asset_file = AssetFileHybridDataset::<S>(PhantomData);
        let dataset_hash = dataset_asset_file.synchronize().await;
        let tracking_content =
            read_json::<AssetFileDirectTreeActionLogsTracking<S>>(self.version_tracking_path())
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
        hash_file(self.actions_file_path()).expect("Target file missing for direct action log")
    }
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        // DirectTreeActionLogStore::<M>::initialize_if_missing(self.file_path())
        SqliteTableArrayStore::<QuestionFlatId<S>, DirectTreeAction<M>>::initialize_if_missing(
            self.actions_file_path(),
        )
        .unwrap_or_else(|e| {
            panic!(
                "Failed to open direct action log sqlite table array store at {}: {}",
                self.actions_file_path(),
                e
            )
        })
    }
}
