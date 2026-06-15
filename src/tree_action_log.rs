use redb::{
    Database, Key as RedbKey, MultimapTableDefinition, ReadableDatabase, ReadableMultimapTable,
    ReadableTable, TableDefinition, TableError as RedbTableError, TypeName, Value as RedbValue,
};
use research_utility::sqlite_table_array_store::SqliteTableArrayStore;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fs, marker::PhantomData, path::PathBuf};
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

const INITIALIZED_KEYS_TABLE_NAME: &str = "initialized_keys";
const ACTION_ROWS_TABLE_NAME: &str = "action_rows";

#[derive(Clone)]
pub struct DirectTreeActionLog<M: LlmModelMarker, S: DatasetSplit> {
    pub question: HybridDatasetQuestion<S>,
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub actions: Vec<DirectTreeAction<M>>,
}

enum ActionLogStoreBackend<M: LlmModelMarker, S: DatasetSplit> {
    Sqlite(SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>>),
    Redb(RedbActionLogStore<M, S>),
}

pub struct ActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    backend: ActionLogStoreBackend<M, S>,
}

impl<M: LlmModelMarker, S: DatasetSplit> ActionLogStore<M, S> {
    pub fn initialize_if_missing(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        Self::initialize_redb_if_missing(db_path)
    }

    pub fn initialize_sqlite_if_missing(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        SqliteTableArrayStore::<QuestionFlatId<S>, DirectTreeAction<M>>::initialize_if_missing(
            db_path,
        )
        .map(|inner| Self {
            backend: ActionLogStoreBackend::Sqlite(inner),
        })
    }

    pub fn initialize_redb_if_missing(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        RedbActionLogStore::<M, S>::initialize_if_missing(db_path).map(|inner| Self {
            backend: ActionLogStoreBackend::Redb(inner),
        })
    }

    pub fn get_keys(&self) -> Result<Vec<QuestionFlatId<S>>, String> {
        match &self.backend {
            ActionLogStoreBackend::Sqlite(store) => store.get_keys(),
            ActionLogStoreBackend::Redb(store) => store.get_keys(),
        }
    }

    pub fn load_table_sorted(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        match &self.backend {
            ActionLogStoreBackend::Sqlite(store) => store.load_table_sorted(table_key),
            ActionLogStoreBackend::Redb(store) => store.load_table_sorted(table_key),
        }
    }

    pub fn load_or_init_table_sorted<F>(
        &self,
        table_key: QuestionFlatId<S>,
        initialize_rows: F,
    ) -> Result<Vec<DirectTreeAction<M>>, String>
    where
        F: FnOnce() -> Vec<(usize, DirectTreeAction<M>)>,
    {
        match &self.backend {
            ActionLogStoreBackend::Sqlite(store) => {
                store.load_or_init_table_sorted(table_key, initialize_rows)
            }
            ActionLogStoreBackend::Redb(store) => {
                store.load_or_init_table_sorted(table_key, initialize_rows)
            }
        }
    }

    pub fn append_at(
        &self,
        table_key: QuestionFlatId<S>,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        match &self.backend {
            ActionLogStoreBackend::Sqlite(store) => store.append_at(table_key, row_index, value),
            ActionLogStoreBackend::Redb(store) => store.append_at(table_key, row_index, value),
        }
    }
}

pub struct DirectTreeActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    // pub metadata_store: SqliteStore<usize, DirectTreeActionLogMetadata>,
    pub action_store: ActionLogStore<M, S>,
    pub _phantom: PhantomData<M>,
}

#[derive(Debug, Clone)]
pub struct ActionStoreAdapter<M: LlmModelMarker, S: DatasetSplit> {
    request_tx: mpsc::UnboundedSender<StoreRequest<M, S>>,
    _phantom: PhantomData<(M, S)>,
}

enum StoreRequest<M: LlmModelMarker, S: DatasetSplit> {
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
impl<M: LlmModelMarker, S: DatasetSplit> ActionStoreAdapter<M, S> {
    pub fn new(store: ActionLogStore<M, S>) -> Self {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        Self::spawn_worker(store, request_rx);
        Self {
            request_tx,
            _phantom: PhantomData,
        }
    }

    fn spawn_worker(
        store: ActionLogStore<M, S>,
        mut request_rx: mpsc::UnboundedReceiver<StoreRequest<M, S>>,
    ) {
        std::thread::Builder::new()
            .name("direct_action_log_store_worker".to_string())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to initialize direct action log store worker runtime");
                runtime.block_on(async move {
                    while let Some(request) = request_rx.recv().await {
                        match request {
                            StoreRequest::GetOrInitActions { key, response_tx } => {
                                let result = store.load_or_init_table_sorted(key, Vec::new);
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
            .expect("failed to spawn direct action log store worker thread");
    }

    pub async fn get_or_init_actions(
        &self,
        key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(StoreRequest::GetOrInitActions { key, response_tx })
            .map_err(|_| "direct action log store worker has shut down".to_string())?;
        response_rx
            .await
            .map_err(|_| "direct action log store worker response dropped".to_string())?
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
            .map_err(|_| "direct action log store worker has shut down".to_string())?;
        response_rx
            .await
            .map_err(|_| "direct action log store worker response dropped".to_string())?
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutPurpose {
    Training,
    Evaluation,
    Testing {},
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
struct StoredActionRow<M: LlmModelMarker> {
    row_index: u64,
    action: DirectTreeAction<M>,
}

impl<M: LlmModelMarker> std::fmt::Debug for StoredActionRow<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredActionRow")
            .field("row_index", &self.row_index)
            .field("action", &"<DirectTreeAction>")
            .finish()
    }
}

impl<M: LlmModelMarker> StoredActionRow<M> {
    fn from_row_index_and_action(
        row_index: usize,
        action: &DirectTreeAction<M>,
    ) -> Result<Self, String> {
        let row_index = u64::try_from(row_index)
            .map_err(|_| format!("row index {row_index} does not fit into u64"))?;
        Ok(Self {
            row_index,
            action: action.clone(),
        })
    }

    fn encode_action(action: &DirectTreeAction<M>) -> Vec<u8> {
        rmp_serde::to_vec_named(action)
            .unwrap_or_else(|e| panic!("failed to serialize DirectTreeAction for redb: {e}"))
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 256);
        bytes.extend_from_slice(&self.row_index.to_be_bytes());
        bytes.extend_from_slice(&Self::encode_action(&self.action));
        bytes
    }

    fn decode(data: &[u8]) -> Self {
        assert!(
            data.len() >= 8,
            "redb StoredActionRow payload must be at least 8 bytes"
        );
        let row_index = u64::from_be_bytes(data[..8].try_into().unwrap());
        let action = rmp_serde::from_slice(&data[8..])
            .unwrap_or_else(|e| panic!("failed to deserialize DirectTreeAction from redb: {e}"));
        Self { row_index, action }
    }
}

impl<M: LlmModelMarker> RedbValue for StoredActionRow<M> {
    type SelfType<'a>
        = StoredActionRow<M>
    where
        Self: 'a;
    type AsBytes<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        Self::decode(data)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        value.encode()
    }

    fn type_name() -> TypeName {
        TypeName::new(&format!(
            "credit_assignment::StoredActionRow<{}>",
            std::any::type_name::<M>()
        ))
    }
}

impl<M: LlmModelMarker> RedbKey for StoredActionRow<M> {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        let ordering = data1[..8].cmp(&data2[..8]);
        if ordering.is_eq() {
            data1[8..].cmp(&data2[8..])
        } else {
            ordering
        }
    }
}

struct RedbActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    db_path: PathBuf,
    db: Database,
    _phantom: PhantomData<(M, S)>,
}

impl<M: LlmModelMarker, S: DatasetSplit> RedbActionLogStore<M, S> {
    fn initialize_if_missing(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        let db_path = db_path.into();
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create parent directory for redb action log store at {}: {}",
                    db_path.display(),
                    e
                )
            })?;
        }
        let db = Database::create(&db_path).map_err(|e| {
            format!(
                "Failed to open redb action log store at {}: {}",
                db_path.display(),
                e
            )
        })?;
        Ok(Self {
            db_path,
            db,
            _phantom: PhantomData,
        })
    }

    fn get_keys(&self) -> Result<Vec<QuestionFlatId<S>>, String> {
        let read_txn = self.db.begin_read().map_err(|e| {
            format!(
                "Failed to begin redb read transaction at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        let table = match read_txn.open_table(initialized_keys_table_def()) {
            Ok(table) => table,
            Err(RedbTableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => {
                return Err(format!(
                    "Failed to open redb initialized keys table at {}: {}",
                    self.db_path.display(),
                    e
                ));
            }
        };

        let mut keys = Vec::new();
        let entries = table.iter().map_err(|e| {
            format!(
                "Failed to iterate redb initialized keys table at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        for entry in entries {
            let (key, _) = entry.map_err(|e| {
                format!(
                    "Failed to read redb initialized key entry at {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
            keys.push(question_flat_id_from_u64::<S>(key.value())?);
        }
        Ok(keys)
    }

    fn load_table_sorted(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        let key_u64 = question_flat_id_to_u64(table_key)?;
        let read_txn = self.db.begin_read().map_err(|e| {
            format!(
                "Failed to begin redb read transaction at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        let initialized_keys = match read_txn.open_table(initialized_keys_table_def()) {
            Ok(table) => table,
            Err(RedbTableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => {
                return Err(format!(
                    "Failed to open redb initialized keys table at {}: {}",
                    self.db_path.display(),
                    e
                ));
            }
        };
        if initialized_keys
            .get(key_u64)
            .map_err(|e| {
                format!(
                    "Failed to fetch redb initialized key {} at {}: {}",
                    key_u64,
                    self.db_path.display(),
                    e
                )
            })?
            .is_none()
        {
            return Ok(Vec::new());
        }

        let action_rows = match read_txn.open_multimap_table(action_rows_table_def::<M>()) {
            Ok(table) => table,
            Err(RedbTableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => {
                return Err(format!(
                    "Failed to open redb action rows table at {}: {}",
                    self.db_path.display(),
                    e
                ));
            }
        };

        let mut actions = Vec::new();
        let values = action_rows.get(key_u64).map_err(|e| {
            format!(
                "Failed to fetch redb action rows for key {} at {}: {}",
                key_u64,
                self.db_path.display(),
                e
            )
        })?;
        for value in values {
            let value = value.map_err(|e| {
                format!(
                    "Failed to decode redb action row for key {} at {}: {}",
                    key_u64,
                    self.db_path.display(),
                    e
                )
            })?;
            let row = value.value();
            actions.push(row.action);
        }
        Ok(actions)
    }

    fn load_or_init_table_sorted<F>(
        &self,
        table_key: QuestionFlatId<S>,
        initialize_rows: F,
    ) -> Result<Vec<DirectTreeAction<M>>, String>
    where
        F: FnOnce() -> Vec<(usize, DirectTreeAction<M>)>,
    {
        let key_u64 = question_flat_id_to_u64(table_key)?;
        let write_txn = self.db.begin_write().map_err(|e| {
            format!(
                "Failed to begin redb write transaction at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        let key_exists = {
            let initialized_keys =
                write_txn
                    .open_table(initialized_keys_table_def())
                    .map_err(|e| {
                        format!(
                            "Failed to open redb initialized keys table at {}: {}",
                            self.db_path.display(),
                            e
                        )
                    })?;
            initialized_keys
                .get(key_u64)
                .map_err(|e| {
                    format!(
                        "Failed to fetch redb initialized key {} at {}: {}",
                        key_u64,
                        self.db_path.display(),
                        e
                    )
                })?
                .is_some()
        };

        if !key_exists {
            {
                let mut initialized_keys = write_txn
                    .open_table(initialized_keys_table_def())
                    .map_err(|e| {
                        format!(
                            "Failed to reopen redb initialized keys table at {}: {}",
                            self.db_path.display(),
                            e
                        )
                    })?;
                initialized_keys.insert(key_u64, 1u8).map_err(|e| {
                    format!(
                        "Failed to insert redb initialized key {} at {}: {}",
                        key_u64,
                        self.db_path.display(),
                        e
                    )
                })?;
            }
            {
                let mut action_rows = write_txn
                    .open_multimap_table(action_rows_table_def::<M>())
                    .map_err(|e| {
                        format!(
                            "Failed to open redb action rows table at {}: {}",
                            self.db_path.display(),
                            e
                        )
                    })?;
                for (row_index, value) in initialize_rows() {
                    self.insert_row_checked(&action_rows, key_u64, row_index, &value)?;
                    action_rows.insert(key_u64, &StoredActionRow::from_row_index_and_action(row_index, &value)?).map_err(|e| {
                        format!(
                            "Failed to insert redb action row for key {} row_index {} at {}: {}",
                            key_u64,
                            row_index,
                            self.db_path.display(),
                            e
                        )
                    })?;
                }
            }
            write_txn.commit().map_err(|e| {
                format!(
                    "Failed to commit redb initialization transaction at {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
        }

        self.load_table_sorted(table_key)
    }

    fn append_at(
        &self,
        table_key: QuestionFlatId<S>,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        let key_u64 = question_flat_id_to_u64(table_key)?;
        let write_txn = self.db.begin_write().map_err(|e| {
            format!(
                "Failed to begin redb write transaction at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        {
            let mut initialized_keys =
                write_txn
                    .open_table(initialized_keys_table_def())
                    .map_err(|e| {
                        format!(
                            "Failed to open redb initialized keys table at {}: {}",
                            self.db_path.display(),
                            e
                        )
                    })?;
            if initialized_keys
                .get(key_u64)
                .map_err(|e| {
                    format!(
                        "Failed to fetch redb initialized key {} at {}: {}",
                        key_u64,
                        self.db_path.display(),
                        e
                    )
                })?
                .is_none()
            {
                initialized_keys.insert(key_u64, 1u8).map_err(|e| {
                    format!(
                        "Failed to insert redb initialized key {} at {}: {}",
                        key_u64,
                        self.db_path.display(),
                        e
                    )
                })?;
            }
        }
        {
            let mut action_rows = write_txn
                .open_multimap_table(action_rows_table_def::<M>())
                .map_err(|e| {
                    format!(
                        "Failed to open redb action rows table at {}: {}",
                        self.db_path.display(),
                        e
                    )
                })?;
            let row = StoredActionRow::from_row_index_and_action(row_index, value)?;
            self.insert_row_checked(&action_rows, key_u64, row_index, value)?;
            action_rows.insert(key_u64, &row).map_err(|e| {
                format!(
                    "Failed to insert redb action row for key {} row_index {} at {}: {}",
                    key_u64,
                    row_index,
                    self.db_path.display(),
                    e
                )
            })?;
        }
        write_txn.commit().map_err(|e| {
            format!(
                "Failed to commit redb action append transaction at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        Ok(())
    }

    fn insert_row_checked(
        &self,
        action_rows: &impl ReadableMultimapTable<u64, StoredActionRow<M>>,
        key_u64: u64,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        let candidate = StoredActionRow::from_row_index_and_action(row_index, value)?;
        let candidate_bytes = candidate.encode();
        let values = action_rows.get(key_u64).map_err(|e| {
            format!(
                "Failed to fetch redb action rows for key {} at {}: {}",
                key_u64,
                self.db_path.display(),
                e
            )
        })?;
        for existing in values {
            let existing = existing.map_err(|e| {
                format!(
                    "Failed to decode redb action row for key {} at {}: {}",
                    key_u64,
                    self.db_path.display(),
                    e
                )
            })?;
            let existing_row = existing.value();
            match existing_row.row_index.cmp(&candidate.row_index) {
                Ordering::Less => continue,
                Ordering::Equal => {
                    if existing_row.encode() == candidate_bytes {
                        return Ok(());
                    }
                    return Err(format!(
                        "Conflicting payload at row_index {} for key {} in redb action log store {}",
                        row_index,
                        key_u64,
                        self.db_path.display()
                    ));
                }
                Ordering::Greater => break,
            }
        }
        Ok(())
    }
}

fn initialized_keys_table_def() -> TableDefinition<'static, u64, u8> {
    TableDefinition::new(INITIALIZED_KEYS_TABLE_NAME)
}

fn action_rows_table_def<M: LlmModelMarker>()
-> MultimapTableDefinition<'static, u64, StoredActionRow<M>> {
    MultimapTableDefinition::new(ACTION_ROWS_TABLE_NAME)
}

fn question_flat_id_to_u64<S: DatasetSplit>(key: QuestionFlatId<S>) -> Result<u64, String> {
    u64::try_from(key.0).map_err(|_| format!("QuestionFlatId {} does not fit into u64", key.0))
}

fn question_flat_id_from_u64<S: DatasetSplit>(value: u64) -> Result<QuestionFlatId<S>, String> {
    let usize_value = usize::try_from(value)
        .map_err(|_| format!("redb QuestionFlatId {value} does not fit into usize"))?;
    Ok(QuestionFlatId(usize_value, PhantomData))
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
) -> ActionLogStore<M, S> {
    ActionLogStore::<M, S>::initialize_if_missing(action_logs_file_path::<M, S>(
        config_nickname,
        epoch,
    ))
    .unwrap_or_else(|e| {
        panic!(
            "Failed to open direct action log store at {}: {}",
            action_logs_file_path::<M, S>(config_nickname, epoch),
            e
        )
    })
}
