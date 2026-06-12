use research_utility::sqlite_table_array_store::SqliteTableArrayStore;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use tokio::sync::{mpsc, oneshot};

use crate::{
    direct_tool::{
        hybrid_dataset::{DatasetSplit, HybridDatasetQuestion, QuestionFlatId},
        posterior_calculation_config::PosteriorCalculationConfig,
        rollout_config::DirectRolloutConfig,
        tree_action::DirectTreeAction,
    },
    jinja_directories::action_logs_path_from_template,
    llm_model::LlmModelMarker,
};

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

pub fn action_logs_file_path<M: LlmModelMarker, S: DatasetSplit>(
    config_nickname: &str,
    epoch: usize,
) -> String {
    action_logs_path_from_template::<S>(M::CLI_NAME, config_nickname, epoch)
        .unwrap_or_else(|err| panic!("Failed to resolve action logs path: {}", err))
}

pub fn open_action_logs<M: LlmModelMarker, S: DatasetSplit>(
    config_nickname: &str,
    epoch: usize,
) -> SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>> {
    SqliteTableArrayStore::<QuestionFlatId<S>, DirectTreeAction<M>>::initialize_if_missing(
        action_logs_file_path::<M, S>(config_nickname, epoch),
    )
    .unwrap_or_else(|e| {
        panic!(
            "Failed to open direct action log sqlite table array store at {}: {}",
            action_logs_file_path::<M, S>(config_nickname, epoch),
            e
        )
    })
}
