use redb::{
    Database, Key as RedbKey, MultimapTableDefinition, ReadableDatabase, ReadableMultimapTable,
    ReadableTable, TableDefinition, TableError as RedbTableError, TypeName, Value as RedbValue,
    WriteTransaction,
};
use research_utility::progress_tui_logger::{log_info, log_warning};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::atomic::AtomicU64,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    direct_tool::{
        hybrid_dataset::{DatasetSplit, HybridDatasetQuestion, QuestionFlatId},
        posterior_calculation_config::PosteriorCalculationConfig,
        rollout_config::DirectRolloutConfig,
        tree_action::DirectTreeAction,
    },
    jinja_directories::action_logs_path_from_template,
    json_toml_utils::write_json,
    llm_model::LlmModelMarker,
};

const INITIALIZED_KEYS_TABLE_NAME: &str = "initialized_keys";
const ACTION_ROWS_TABLE_NAME: &str = "action_rows";
const SORT_GENERATIONS_DIR_NAME: &str = "sorted_generations";
const ACTIVE_SORT_GENERATION_FILE_NAME: &str = "sorted_generation.txt";
const SORT_TEMP_FILE_PREFIX: &str = "sort_tmp_";

static SORT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn action_log_config_bundle_path(base_path: &Path) -> PathBuf {
    if base_path.extension().and_then(|ext| ext.to_str()) == Some("extsort") {
        base_path.join("config_bundle.json")
    } else {
        base_path.with_extension("config_bundle.json")
    }
}

pub fn action_log_config_bundle_file_path(action_logs_path: impl AsRef<Path>) -> PathBuf {
    action_log_config_bundle_path(action_logs_path.as_ref())
}

#[derive(Clone)]
pub struct DirectTreeActionLog<M: LlmModelMarker, S: DatasetSplit> {
    pub question: HybridDatasetQuestion<S>,
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub actions: Vec<DirectTreeAction<M>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct ActionLogConfigBundle<S: DatasetSplit> {
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
}

pub struct ActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    backend: ActionLogStoreBackend<M, S>,
}

#[allow(dead_code)]
enum ActionLogStoreBackend<M: LlmModelMarker, S: DatasetSplit> {
    Redb(RedbActionLogStore<M, S>),
    ExtSort(ExtSortActionLogStore<M, S>),
}

impl<M: LlmModelMarker, S: DatasetSplit> ActionLogStore<M, S> {
    pub fn initialize_if_missing(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        ExtSortActionLogStore::<M, S>::initialize_if_missing(db_path).map(|store| Self {
            backend: ActionLogStoreBackend::ExtSort(store),
        })
    }

    pub fn write_config_bundle_if_missing(
        &self,
        config_bundle: &ActionLogConfigBundle<S>,
    ) -> Result<(), String> {
        match &self.backend {
            ActionLogStoreBackend::Redb(store) => {
                store.write_config_bundle_if_missing(config_bundle)
            }
            ActionLogStoreBackend::ExtSort(store) => {
                store.write_config_bundle_if_missing(config_bundle)
            }
        }
    }

    pub fn get_keys(&self) -> Result<Vec<QuestionFlatId<S>>, String> {
        match &self.backend {
            ActionLogStoreBackend::Redb(store) => store.get_keys(),
            ActionLogStoreBackend::ExtSort(store) => store.get_keys(),
        }
    }

    pub fn load_action_log(
        &self,
        question_flat_id: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        match &self.backend {
            ActionLogStoreBackend::Redb(store) => store.load_action_log(question_flat_id),
            ActionLogStoreBackend::ExtSort(store) => store.load_action_log(question_flat_id),
        }
    }

    pub fn load_or_init_action_log(
        &self,
        question_flat_id: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        match &self.backend {
            ActionLogStoreBackend::Redb(store) => store.load_or_init_action_log(question_flat_id),
            ActionLogStoreBackend::ExtSort(store) => {
                store.load_or_init_action_log(question_flat_id)
            }
        }
    }

    pub fn append(
        &self,
        question_flat_id: QuestionFlatId<S>,
        action_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        match &self.backend {
            ActionLogStoreBackend::Redb(store) => {
                store.append(question_flat_id, action_index, value)
            }
            ActionLogStoreBackend::ExtSort(store) => {
                store.append(question_flat_id, action_index, value)
            }
        }
    }

    pub fn sort(&self) -> Result<(), String> {
        match &self.backend {
            ActionLogStoreBackend::Redb(store) => store.sort(),
            ActionLogStoreBackend::ExtSort(store) => store.sort(),
        }
    }

    pub fn load_table_sorted(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        self.load_action_log(table_key)
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
            ActionLogStoreBackend::Redb(store) => {
                store.load_or_init_table_sorted(table_key, initialize_rows)
            }
            ActionLogStoreBackend::ExtSort(store) => {
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
        self.append(table_key, row_index, value)
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

// Redb batching model: keep one live write transaction open, apply writes directly into it,
// and only call commit() on timed flushes or at the end of rollout_all().
#[allow(dead_code)]
struct RedbInlineActionStoreState<M: LlmModelMarker, S: DatasetSplit> {
    store: RedbActionLogStore<M, S>,
    write_txn: Option<WriteTransaction>,
    has_pending_writes: bool,
}

#[allow(dead_code)]
impl<M: LlmModelMarker, S: DatasetSplit> RedbInlineActionStoreState<M, S> {
    fn new(store: RedbActionLogStore<M, S>) -> Self {
        log_info(format!(
            "RedbInlineActionStoreState::new: acquiring write transaction lock on {}",
            store.db_path.display()
        ));
        let write_txn = store
            .begin_write_txn()
            .unwrap_or_else(|err| panic!("failed to initialize redb write transaction: {err}"));
        log_info(format!(
            "RedbInlineActionStoreState::new: write transaction lock acquired on {}",
            store.db_path.display()
        ));
        Self {
            store,
            write_txn: Some(write_txn),
            has_pending_writes: false,
        }
    }

    fn current_write_txn(&self) -> Result<&WriteTransaction, String> {
        self.write_txn
            .as_ref()
            .ok_or_else(|| "redb write transaction is not initialized".to_string())
    }

    fn get_or_init_actions(
        &mut self,
        key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        let (actions, did_write) = self.store.load_or_init_table_sorted_in_write_txn(
            self.current_write_txn()?,
            key,
            Vec::new,
        )?;
        self.has_pending_writes |= did_write;
        Ok(actions)
    }

    fn append_action_at(
        &mut self,
        key: QuestionFlatId<S>,
        action_index: usize,
        action: &DirectTreeAction<M>,
        commit_after_write: bool,
    ) -> Result<bool, String> {
        let did_write = self.store.append_at_in_write_txn(
            self.current_write_txn()?,
            key,
            action_index,
            action,
        )?;
        self.has_pending_writes |= did_write;
        if commit_after_write {
            self.commit_pending()
        } else {
            Ok(false)
        }
    }

    fn commit_pending(&mut self) -> Result<bool, String> {
        if !self.has_pending_writes {
            return Ok(false);
        }
        let write_txn = self
            .write_txn
            .take()
            .ok_or_else(|| "redb write transaction is not initialized".to_string())?;
        self.store.commit_write_txn(write_txn)?;
        self.write_txn = Some(self.store.begin_write_txn()?);
        self.has_pending_writes = false;
        Ok(true)
    }
}

impl<M: LlmModelMarker, S: DatasetSplit> Drop for RedbInlineActionStoreState<M, S> {
    fn drop(&mut self) {
        log_info(format!(
            "RedbInlineActionStoreState::drop: releasing write transaction lock on {}",
            self.store.db_path.display()
        ));
        // Explicitly abort the write transaction to release the file lock.
        // redb's WriteTransaction::drop does call abort_inner(), but only when
        // the transaction hasn't been completed and the thread isn't panicking.
        // Explicitly dropping here ensures the lock is released promptly and
        // avoids any edge cases where the lock might persist.
        drop(self.write_txn.take());
        log_info(format!(
            "RedbInlineActionStoreState::drop: write transaction lock released on {}",
            self.store.db_path.display()
        ));
    }
}

#[allow(dead_code)]
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

    fn write_config_bundle_if_missing(
        &self,
        config_bundle: &ActionLogConfigBundle<S>,
    ) -> Result<(), String> {
        let bundle_path = action_log_config_bundle_path(&self.db_path);
        if bundle_path.exists() {
            return Ok(());
        }
        write_json(bundle_path, config_bundle)
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
        self.load_table_state(table_key).map(|(_, actions)| actions)
    }

    fn load_action_log(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        self.load_table_sorted(table_key).and_then(|actions| {
            if actions.is_empty() {
                Err(format!(
                    "Action store at {} is missing key {}; a key is supposed to exist but no corresponding action is found",
                    self.db_path.display(),
                    question_flat_id_to_u64(table_key)?
                ))
            } else {
                Ok(actions)
            }
        })
    }

    fn load_or_init_action_log(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        self.load_table_sorted(table_key)
    }

    fn load_table_state(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<(bool, Vec<DirectTreeAction<M>>), String> {
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
            Err(RedbTableError::TableDoesNotExist(_)) => return Ok((false, Vec::new())),
            Err(e) => {
                return Err(format!(
                    "Failed to open redb initialized keys table at {}: {}",
                    self.db_path.display(),
                    e
                ));
            }
        };
        let exists = initialized_keys
            .get(key_u64)
            .map_err(|e| {
                format!(
                    "Failed to fetch redb initialized key {} at {}: {}",
                    key_u64,
                    self.db_path.display(),
                    e
                )
            })?
            .is_some();
        if !exists {
            return Ok((false, Vec::new()));
        }

        let action_rows = match read_txn.open_multimap_table(action_rows_table_def::<M>()) {
            Ok(table) => table,
            Err(RedbTableError::TableDoesNotExist(_)) => return Ok((true, Vec::new())),
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
        Ok((true, actions))
    }

    fn load_or_init_table_sorted<F>(
        &self,
        table_key: QuestionFlatId<S>,
        initialize_rows: F,
    ) -> Result<Vec<DirectTreeAction<M>>, String>
    where
        F: FnOnce() -> Vec<(usize, DirectTreeAction<M>)>,
    {
        let write_txn = self.begin_write_txn()?;
        let (actions, did_write) =
            self.load_or_init_table_sorted_in_write_txn(&write_txn, table_key, initialize_rows)?;
        if did_write {
            self.commit_write_txn(write_txn)?;
        }
        Ok(actions)
    }

    fn append_at(
        &self,
        table_key: QuestionFlatId<S>,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        let write_txn = self.begin_write_txn()?;
        let did_write = self.append_at_in_write_txn(&write_txn, table_key, row_index, value)?;
        if did_write {
            self.commit_write_txn(write_txn)?;
        }
        Ok(())
    }

    fn append(
        &self,
        table_key: QuestionFlatId<S>,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        self.append_at(table_key, row_index, value)
    }

    fn sort(&self) -> Result<(), String> {
        Ok(())
    }

    fn begin_write_txn(&self) -> Result<WriteTransaction, String> {
        self.db.begin_write().map_err(|e| {
            format!(
                "Failed to begin redb write transaction at {}: {}",
                self.db_path.display(),
                e
            )
        })
    }

    fn commit_write_txn(&self, write_txn: WriteTransaction) -> Result<(), String> {
        write_txn.commit().map_err(|e| {
            format!(
                "Failed to commit redb action transaction at {}: {}",
                self.db_path.display(),
                e
            )
        })
    }

    fn load_or_init_table_sorted_in_write_txn<F>(
        &self,
        write_txn: &WriteTransaction,
        table_key: QuestionFlatId<S>,
        initialize_rows: F,
    ) -> Result<(Vec<DirectTreeAction<M>>, bool), String>
    where
        F: FnOnce() -> Vec<(usize, DirectTreeAction<M>)>,
    {
        let key_u64 = question_flat_id_to_u64(table_key)?;
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

        let mut did_write = false;
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
                    action_rows.insert(
                        key_u64,
                        &StoredActionRow::from_row_index_and_action(row_index, &value)?,
                    ).map_err(|e| {
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
            did_write = true;
        }

        let action_rows = write_txn
            .open_multimap_table(action_rows_table_def::<M>())
            .map_err(|e| {
                format!(
                    "Failed to open redb action rows table at {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
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
            actions.push(value.value().action);
        }
        Ok((actions, did_write))
    }

    fn append_at_in_write_txn(
        &self,
        write_txn: &WriteTransaction,
        table_key: QuestionFlatId<S>,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<bool, String> {
        let key_u64 = question_flat_id_to_u64(table_key)?;
        let mut did_write = false;
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
                did_write = true;
            }
        }
        let mut action_rows = write_txn
            .open_multimap_table(action_rows_table_def::<M>())
            .map_err(|e| {
                format!(
                    "Failed to open redb action rows table at {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
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
                        return Ok(did_write);
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
        action_rows.insert(key_u64, &candidate).map_err(|e| {
            format!(
                "Failed to insert redb action row for key {} row_index {} at {}: {}",
                key_u64,
                row_index,
                self.db_path.display(),
                e
            )
        })?;
        Ok(true)
    }
}

#[allow(dead_code)]
struct ExtSortActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    base_path: PathBuf,
    storage_dir: PathBuf,
    _phantom: PhantomData<(M, S)>,
}

#[derive(Clone, Debug)]
struct ExtSortActionRow {
    question_flat_id: u64,
    row_index: u64,
    action_payload: Vec<u8>,
}

impl ExtSortActionRow {
    fn from_row_index_and_action<M: LlmModelMarker>(
        question_flat_id: u64,
        row_index: usize,
        action: &DirectTreeAction<M>,
    ) -> Result<Self, String> {
        let row_index = u64::try_from(row_index)
            .map_err(|_| format!("row index {row_index} does not fit into u64"))?;
        let action_payload = bincode::serialize(action).unwrap_or_else(|e| {
            panic!("failed to serialize DirectTreeAction for extsort with bincode: {e}")
        });
        Ok(Self {
            question_flat_id,
            row_index,
            action_payload,
        })
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24 + self.action_payload.len());
        bytes.extend_from_slice(&self.question_flat_id.to_be_bytes());
        bytes.extend_from_slice(&self.row_index.to_be_bytes());
        bytes.extend_from_slice(&(self.action_payload.len() as u64).to_be_bytes());
        bytes.extend_from_slice(&self.action_payload);
        bytes
    }

    fn decode(reader: &mut impl Read) -> std::io::Result<Self> {
        let mut question_flat_id_bytes = [0u8; 8];
        reader.read_exact(&mut question_flat_id_bytes)?;
        let mut row_index_bytes = [0u8; 8];
        reader.read_exact(&mut row_index_bytes)?;
        let mut payload_len_bytes = [0u8; 8];
        reader.read_exact(&mut payload_len_bytes)?;
        let payload_len = u64::from_be_bytes(payload_len_bytes) as usize;
        let mut action_payload = vec![0u8; payload_len];
        reader.read_exact(&mut action_payload)?;
        Ok(Self {
            question_flat_id: u64::from_be_bytes(question_flat_id_bytes),
            row_index: u64::from_be_bytes(row_index_bytes),
            action_payload,
        })
    }

    fn decode_action<M: LlmModelMarker>(&self) -> Result<DirectTreeAction<M>, String> {
        bincode::deserialize(&self.action_payload).map_err(|e| {
            format!("failed to deserialize DirectTreeAction from extsort bincode payload: {e}")
        })
    }
}

impl extsort::Sortable for ExtSortActionRow {
    fn encode<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let bytes = self.encode();
        writer.write_all(&bytes)
    }

    fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        Self::decode(reader)
    }
}

#[derive(Clone, Copy, Debug)]
struct ExtSortKeyRecord {
    question_flat_id: u64,
}

#[derive(Debug, Clone)]
struct ResolvedSortArtifactPaths {
    actions: PathBuf,
    index: PathBuf,
    source_len: PathBuf,
}

#[derive(Debug, Clone)]
struct SortGenerationCandidate {
    token: String,
    modified: SystemTime,
}

impl ExtSortKeyRecord {
    fn encode(&self) -> [u8; 8] {
        self.question_flat_id.to_be_bytes()
    }

    fn decode(reader: &mut impl Read) -> std::io::Result<Option<Self>> {
        let mut bytes = [0u8; 8];
        match reader.read_exact(&mut bytes) {
            Ok(()) => Ok(Some(Self {
                question_flat_id: u64::from_be_bytes(bytes),
            })),
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(err) => Err(err),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ExtSortIndexRecord {
    question_flat_id: u64,
    start_offset: u64,
    action_count: u64,
}

impl ExtSortIndexRecord {
    fn encode(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        bytes[..8].copy_from_slice(&self.question_flat_id.to_be_bytes());
        bytes[8..16].copy_from_slice(&self.start_offset.to_be_bytes());
        bytes[16..24].copy_from_slice(&self.action_count.to_be_bytes());
        bytes
    }

    fn decode(reader: &mut impl Read) -> std::io::Result<Option<Self>> {
        let mut bytes = [0u8; 24];
        match reader.read_exact(&mut bytes) {
            Ok(()) => Ok(Some(Self {
                question_flat_id: u64::from_be_bytes(bytes[..8].try_into().unwrap()),
                start_offset: u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
                action_count: u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
            })),
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(err) => Err(err),
        }
    }
}

#[allow(dead_code)]
impl<M: LlmModelMarker, S: DatasetSplit> ExtSortActionLogStore<M, S> {
    fn initialize_if_missing(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        let base_path = db_path.into();
        if let Some(parent) = base_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create parent directory for extsort action log store at {}: {}",
                    base_path.display(),
                    e
                )
            })?;
        }
        let storage_dir = extsort_storage_dir(&base_path);
        fs::create_dir_all(&storage_dir).map_err(|e| {
            format!(
                "Failed to create extsort action log storage directory at {}: {}",
                storage_dir.display(),
                e
            )
        })?;
        let store = Self {
            base_path,
            storage_dir,
            _phantom: PhantomData,
        };
        store.repair_sort_generation_state()?;
        Ok(store)
    }

    fn write_config_bundle_if_missing(
        &self,
        config_bundle: &ActionLogConfigBundle<S>,
    ) -> Result<(), String> {
        let bundle_path = action_log_config_bundle_path(&self.base_path);
        if bundle_path.exists() {
            return Ok(());
        }
        write_json(bundle_path, config_bundle)
    }

    fn get_keys(&self) -> Result<Vec<QuestionFlatId<S>>, String> {
        let index = self.read_index_map()?;
        index.into_keys().map(question_flat_id_from_u64).collect()
    }

    fn load_action_log(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        let key_u64 = question_flat_id_to_u64(table_key)?;
        let paths = self.resolved_sort_artifact_paths()?;
        let index = self.read_index_map()?;
        let Some(entry) = index.get(&key_u64) else {
            return Err(format!(
                "Action store at {} is missing key {}; a key is supposed to exist but no corresponding action is found",
                self.base_path.display(),
                key_u64
            ));
        };
        let file = File::open(&paths.actions).map_err(|e| {
            format!(
                "Failed to open extsort actions file at {}: {}",
                paths.actions.display(),
                e
            )
        })?;
        let mut reader = BufReader::new(file);
        reader
            .seek(SeekFrom::Start(entry.start_offset))
            .map_err(|e| {
                format!(
                    "Failed to seek extsort actions file at {}: {}",
                    paths.actions.display(),
                    e
                )
            })?;
        let mut actions = Vec::with_capacity(entry.action_count as usize);
        for expected_index in 0..entry.action_count {
            let row = ExtSortActionRow::decode(&mut reader).map_err(|e| {
                format!(
                    "Failed to decode extsort action row for key {} at {}: {}",
                    key_u64,
                    paths.actions.display(),
                    e
                )
            })?;
            assert_eq!(
                row.question_flat_id, key_u64,
                "load_action_log must read a contiguous block for one question_flat_id"
            );
            assert_eq!(
                row.row_index, expected_index,
                "action indices must be sequential starting at 0 for question_flat_id {}",
                key_u64
            );
            actions.push(row.decode_action::<M>()?);
        }
        let mut next_header = [0u8; 8];
        match reader.read_exact(&mut next_header) {
            Ok(()) => {
                let next_key = u64::from_be_bytes(next_header);
                assert_ne!(
                    next_key, key_u64,
                    "load_action_log must stop before the next question_flat_id"
                );
            }
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {}
            Err(err) => {
                return Err(format!(
                    "Failed to probe for the next extsort action row at {}: {}",
                    paths.actions.display(),
                    err
                ));
            }
        }
        Ok(actions)
    }

    fn load_or_init_action_log(
        &self,
        table_key: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        let key_u64 = question_flat_id_to_u64(table_key)?;
        if self.read_index_map()?.contains_key(&key_u64) {
            return self.load_action_log(table_key);
        }
        Ok(Vec::new())
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
        if self.key_exists(key_u64)? {
            return self.load_action_log(table_key);
        }

        self.append_initialized_key(key_u64)?;
        let mut initialized_rows = initialize_rows();
        initialized_rows.sort_by_key(|(row_index, _)| *row_index);
        let actions = initialized_rows
            .iter()
            .map(|(_, action)| action.clone())
            .collect::<Vec<_>>();
        for (row_index, action) in initialized_rows {
            self.append_action_row(key_u64, row_index, &action)?;
        }
        Ok(actions)
    }

    fn append(
        &self,
        table_key: QuestionFlatId<S>,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        let key_u64 = question_flat_id_to_u64(table_key)?;
        self.append_action_row(key_u64, row_index, value)
    }

    fn sort(&self) -> Result<(), String> {
        let source_len = self.file_len(self.pending_actions_path())?;
        if self.is_current_generation_sorted_for_source_len(source_len)? {
            return Ok(());
        }

        log_info("Sorting action store...".to_string());
        let pending_rows = self.read_pending_action_rows()?;
        let sorter = extsort::ExternalSorter::new();
        let sorted_iter = sorter
            .sort_by_key(pending_rows, |row| (row.question_flat_id, row.row_index))
            .map_err(|e| {
                format!(
                    "Failed to external-sort action rows for extsort action log store at {}: {}",
                    self.base_path.display(),
                    e
                )
            })?;
        let sorted_rows = sorted_iter
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(|e| {
                format!(
                    "Failed to collect sorted extsort action rows for store at {}: {}",
                    self.base_path.display(),
                    e
                )
            })?;

        let publish_token = self.new_sort_publish_token(source_len);
        let generation_dir = self.sort_generation_dir(&publish_token);
        fs::create_dir_all(&generation_dir).map_err(|e| {
            format!(
                "Failed to create extsort generation directory at {}: {}",
                generation_dir.display(),
                e
            )
        })?;

        let sorted_actions_path = self.sorted_actions_path_for_generation(&publish_token);
        let index_path = self.index_path_for_generation(&publish_token);
        let source_len_path = self.sorted_source_len_path_for_generation(&publish_token);
        let active_generation_temp_path = self.active_sort_generation_temp_path(&publish_token);

        let sorted_actions_file = File::create(&sorted_actions_path).map_err(|e| {
            format!(
                "Failed to create sorted extsort actions file at {}: {}",
                sorted_actions_path.display(),
                e
            )
        })?;
        let index_file = File::create(&index_path).map_err(|e| {
            format!(
                "Failed to create extsort index file at {}: {}",
                index_path.display(),
                e
            )
        })?;
        let mut actions_writer = BufWriter::new(sorted_actions_file);
        let mut index_writer = BufWriter::new(index_file);
        let mut rows_iter = sorted_rows.into_iter().peekable();
        let mut current_offset = 0u64;
        while let Some(first_row) = rows_iter.peek() {
            let key = first_row.question_flat_id;
            let start_offset = current_offset;
            let mut action_count = 0u64;
            let mut expected_index = 0u64;
            while let Some(row) = rows_iter.peek() {
                if row.question_flat_id != key {
                    break;
                }
                let row = rows_iter.next().expect("peeked row must exist");
                assert_eq!(
                    row.row_index, expected_index,
                    "action indices must be sequential for question_flat_id {}",
                    key
                );
                let encoded = row.encode();
                current_offset +=
                    u64::try_from(encoded.len()).expect("encoded row length must fit in u64");
                actions_writer.write_all(&encoded).map_err(|e| {
                    format!(
                        "Failed to write sorted extsort action row at {}: {}",
                        sorted_actions_path.display(),
                        e
                    )
                })?;
                action_count += 1;
                expected_index += 1;
            }
            index_writer
                .write_all(
                    &ExtSortIndexRecord {
                        question_flat_id: key,
                        start_offset,
                        action_count,
                    }
                    .encode(),
                )
                .map_err(|e| {
                    format!(
                        "Failed to write extsort index entry at {}: {}",
                        index_path.display(),
                        e
                    )
                })?;
        }

        actions_writer.flush().map_err(|e| {
            format!(
                "Failed to flush sorted extsort actions file at {}: {}",
                sorted_actions_path.display(),
                e
            )
        })?;
        actions_writer.get_ref().sync_all().map_err(|e| {
            format!(
                "Failed to sync sorted extsort actions file at {}: {}",
                sorted_actions_path.display(),
                e
            )
        })?;
        index_writer.flush().map_err(|e| {
            format!(
                "Failed to flush extsort index file at {}: {}",
                index_path.display(),
                e
            )
        })?;
        index_writer.get_ref().sync_all().map_err(|e| {
            format!(
                "Failed to sync extsort index file at {}: {}",
                index_path.display(),
                e
            )
        })?;
        self.write_sorted_source_len_to_path(&source_len_path, source_len)?;

        let current_source_len = self.file_len(self.pending_actions_path())?;
        if current_source_len != source_len {
            let _ = self.remove_sort_generation(&generation_dir);
            return Err(format!(
                "Action store at {} changed while sorting: pending actions length was {} at start but {} before publish",
                self.base_path.display(),
                source_len,
                current_source_len
            ));
        }

        self.write_sort_generation_token_to_path(&active_generation_temp_path, &publish_token)?;
        self.publish_sort_generation(&active_generation_temp_path)?;
        if let Err(err) = self.cleanup_sort_artifacts(Some(&publish_token)) {
            log_warning(format!(
                "Completed extsort publish for {} but cleanup of obsolete artifacts failed: {}",
                self.base_path.display(),
                err
            ));
        }
        log_info("Action store sorted.".to_string());
        Ok(())
    }

    fn key_exists(&self, key_u64: u64) -> Result<bool, String> {
        if self
            .read_index_keys()?
            .into_iter()
            .any(|key| key == key_u64)
        {
            return Ok(true);
        }
        Ok(self
            .read_pending_keys()?
            .into_iter()
            .any(|key| key == key_u64))
    }

    fn append_initialized_key(&self, key_u64: u64) -> Result<(), String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.pending_keys_path())
            .map_err(|e| {
                format!(
                    "Failed to open pending extsort keys file at {}: {}",
                    self.pending_keys_path().display(),
                    e
                )
            })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(
                &ExtSortKeyRecord {
                    question_flat_id: key_u64,
                }
                .encode(),
            )
            .map_err(|e| {
                format!(
                    "Failed to append extsort initialized key {} at {}: {}",
                    key_u64,
                    self.pending_keys_path().display(),
                    e
                )
            })?;
        writer.flush().map_err(|e| {
            format!(
                "Failed to flush extsort pending keys file at {}: {}",
                self.pending_keys_path().display(),
                e
            )
        })
    }

    fn append_action_row(
        &self,
        key_u64: u64,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        let row = ExtSortActionRow::from_row_index_and_action(key_u64, row_index, value)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.pending_actions_path())
            .map_err(|e| {
                format!(
                    "Failed to open pending extsort actions file at {}: {}",
                    self.pending_actions_path().display(),
                    e
                )
            })?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&row.encode()).map_err(|e| {
            format!(
                "Failed to append extsort action row for key {} row_index {} at {}: {}",
                key_u64,
                row_index,
                self.pending_actions_path().display(),
                e
            )
        })?;
        writer.flush().map_err(|e| {
            format!(
                "Failed to flush pending extsort actions file at {}: {}",
                self.pending_actions_path().display(),
                e
            )
        })
    }

    fn read_pending_keys(&self) -> Result<Vec<u64>, String> {
        let mut keys = Vec::new();
        let path = self.pending_keys_path();
        if !path.exists() {
            return Ok(keys);
        }
        let file = File::open(&path).map_err(|e| {
            format!(
                "Failed to open pending extsort keys file at {}: {}",
                path.display(),
                e
            )
        })?;
        let mut reader = BufReader::new(file);
        while let Some(record) = ExtSortKeyRecord::decode(&mut reader).map_err(|e| {
            format!(
                "Failed to decode pending extsort key record at {}: {}",
                path.display(),
                e
            )
        })? {
            keys.push(record.question_flat_id);
        }
        Ok(keys)
    }

    fn read_index_keys(&self) -> Result<Vec<u64>, String> {
        let mut keys = Vec::new();
        let paths = self.resolved_sort_artifact_paths()?;
        let path = paths.index;
        if !path.exists() {
            return Ok(keys);
        }
        let file = File::open(&path).map_err(|e| {
            format!(
                "Failed to open extsort index file at {}: {}",
                path.display(),
                e
            )
        })?;
        let mut reader = BufReader::new(file);
        while let Some(record) = ExtSortIndexRecord::decode(&mut reader).map_err(|e| {
            format!(
                "Failed to decode extsort index record at {}: {}",
                path.display(),
                e
            )
        })? {
            keys.push(record.question_flat_id);
        }
        Ok(keys)
    }

    fn read_index_map(&self) -> Result<BTreeMap<u64, ExtSortIndexRecord>, String> {
        let mut index = BTreeMap::new();
        let paths = self.resolved_sort_artifact_paths()?;
        let path = paths.index;
        if !path.exists() {
            return Ok(index);
        }
        let file = File::open(&path).map_err(|e| {
            format!(
                "Failed to open extsort index file at {}: {}",
                path.display(),
                e
            )
        })?;
        let mut reader = BufReader::new(file);
        while let Some(record) = ExtSortIndexRecord::decode(&mut reader).map_err(|e| {
            format!(
                "Failed to decode extsort index record at {}: {}",
                path.display(),
                e
            )
        })? {
            index.insert(record.question_flat_id, record);
        }
        Ok(index)
    }

    fn read_pending_action_rows(&self) -> Result<Vec<ExtSortActionRow>, String> {
        let mut rows = Vec::new();
        let path = self.pending_actions_path();
        if !path.exists() {
            return Ok(rows);
        }
        let file = File::open(&path).map_err(|e| {
            format!(
                "Failed to open pending extsort actions file at {}: {}",
                path.display(),
                e
            )
        })?;
        let mut reader = BufReader::new(file);
        loop {
            match ExtSortActionRow::decode(&mut reader) {
                Ok(row) => rows.push(row),
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(err) => {
                    return Err(format!(
                        "Failed to decode pending extsort action row at {}: {}",
                        path.display(),
                        err
                    ));
                }
            }
        }
        Ok(rows)
    }

    fn file_len(&self, path: PathBuf) -> Result<u64, String> {
        match fs::metadata(&path) {
            Ok(metadata) => Ok(metadata.len()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(err) => Err(format!("Failed to stat {}: {}", path.display(), err)),
        }
    }

    fn read_sorted_source_len(&self) -> Result<Option<u64>, String> {
        let paths = self.resolved_sort_artifact_paths()?;
        let path = paths.source_len;
        if !path.exists() {
            return Ok(None);
        }
        let contents = fs::read_to_string(&path).map_err(|e| {
            format!(
                "Failed to read sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })?;
        let trimmed = contents.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        trimmed.parse::<u64>().map(Some).map_err(|e| {
            format!(
                "Failed to parse sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })
    }

    fn write_sorted_source_len(&self, len: u64) -> Result<(), String> {
        fs::write(self.legacy_sorted_source_len_path(), len.to_string()).map_err(|e| {
            format!(
                "Failed to write sorted source length marker at {}: {}",
                self.legacy_sorted_source_len_path().display(),
                e
            )
        })
    }

    fn write_sorted_source_len_to_path(&self, path: &Path, len: u64) -> Result<(), String> {
        let file = File::create(path).map_err(|e| {
            format!(
                "Failed to create sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })?;
        let mut writer = BufWriter::new(file);
        write!(writer, "{}", len).map_err(|e| {
            format!(
                "Failed to write sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })?;
        writer.flush().map_err(|e| {
            format!(
                "Failed to flush sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })?;
        writer.get_ref().sync_all().map_err(|e| {
            format!(
                "Failed to sync sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })
    }

    fn write_sort_generation_token_to_path(&self, path: &Path, token: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| {
            format!(
                "Failed to create sorted generation marker at {}: {}",
                path.display(),
                e
            )
        })?;
        let mut writer = BufWriter::new(file);
        write!(writer, "{}", token).map_err(|e| {
            format!(
                "Failed to write sorted generation marker at {}: {}",
                path.display(),
                e
            )
        })?;
        writer.flush().map_err(|e| {
            format!(
                "Failed to flush sorted generation marker at {}: {}",
                path.display(),
                e
            )
        })?;
        writer.get_ref().sync_all().map_err(|e| {
            format!(
                "Failed to sync sorted generation marker at {}: {}",
                path.display(),
                e
            )
        })
    }

    fn publish_sort_generation(&self, temp_generation_marker_path: &Path) -> Result<(), String> {
        fs::rename(
            temp_generation_marker_path,
            self.active_sort_generation_path(),
        )
        .map_err(|e| {
            format!(
                "Failed to publish sorted generation marker from {} to {}: {}",
                temp_generation_marker_path.display(),
                self.active_sort_generation_path().display(),
                e
            )
        })?;
        self.sync_directory(&self.storage_dir)?;
        Ok(())
    }

    fn remove_sort_generation(&self, generation_dir: &Path) -> Result<(), String> {
        match fs::remove_dir_all(generation_dir) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(format!(
                "Failed to remove sorted generation directory at {}: {}",
                generation_dir.display(),
                err
            )),
        }
    }

    fn read_active_sort_generation_token(&self) -> Result<Option<String>, String> {
        let path = self.active_sort_generation_path();
        if !path.exists() {
            return Ok(None);
        }
        let contents = fs::read_to_string(&path).map_err(|e| {
            format!(
                "Failed to read sorted generation marker at {}: {}",
                path.display(),
                e
            )
        })?;
        let token = contents.trim();
        if token.is_empty() {
            return Ok(None);
        }
        Ok(Some(token.to_string()))
    }

    fn is_current_generation_sorted_for_source_len(&self, source_len: u64) -> Result<bool, String> {
        let Some(candidate) = self.resolve_current_sort_generation_candidate()? else {
            return Ok(false);
        };
        let paths = self.sort_generation_artifact_paths(&candidate.token);
        let marker_len = self.read_sorted_source_len_from_path(&paths.source_len)?;
        Ok(marker_len == Some(source_len) && paths.actions.exists() && paths.index.exists())
    }

    fn resolved_sort_artifact_paths(&self) -> Result<ResolvedSortArtifactPaths, String> {
        if let Some(candidate) = self.resolve_current_sort_generation_candidate()? {
            let paths = self.sort_generation_artifact_paths(&candidate.token);
            if !paths.actions.exists() || !paths.index.exists() || !paths.source_len.exists() {
                return Err(format!(
                    "Sorted generation {} is active for {} but the expected files are missing",
                    candidate.token,
                    self.storage_dir.display()
                ));
            }
            return Ok(paths);
        }
        Ok(ResolvedSortArtifactPaths {
            actions: self.legacy_sorted_actions_path(),
            index: self.legacy_index_path(),
            source_len: self.legacy_sorted_source_len_path(),
        })
    }

    fn read_sorted_source_len_from_path(&self, path: &Path) -> Result<Option<u64>, String> {
        if !path.exists() {
            return Ok(None);
        }
        let contents = fs::read_to_string(path).map_err(|e| {
            format!(
                "Failed to read sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })?;
        let trimmed = contents.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        trimmed.parse::<u64>().map(Some).map_err(|e| {
            format!(
                "Failed to parse sorted source length marker at {}: {}",
                path.display(),
                e
            )
        })
    }

    fn is_generation_complete(&self, token: &str) -> Result<bool, String> {
        let paths = self.sort_generation_artifact_paths(token);
        if !paths.actions.exists() || !paths.index.exists() || !paths.source_len.exists() {
            return Ok(false);
        }
        Ok(self
            .read_sorted_source_len_from_path(&paths.source_len)?
            .is_some())
    }

    fn complete_generation_candidate(
        &self,
        token: &str,
    ) -> Result<Option<SortGenerationCandidate>, String> {
        if !self.is_generation_complete(token)? {
            return Ok(None);
        }
        let generation_dir = self.sort_generation_dir(token);
        let modified = fs::metadata(&generation_dir)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        Ok(Some(SortGenerationCandidate {
            token: token.to_string(),
            modified,
        }))
    }

    fn scan_latest_complete_generation_candidate(
        &self,
    ) -> Result<Option<SortGenerationCandidate>, String> {
        let generations_dir = self.sort_generations_dir();
        if !generations_dir.exists() {
            return Ok(None);
        }
        let mut best: Option<SortGenerationCandidate> = None;
        for entry in fs::read_dir(&generations_dir).map_err(|e| {
            format!(
                "Failed to read extsort generations directory at {}: {}",
                generations_dir.display(),
                e
            )
        })? {
            let entry = entry.map_err(|e| {
                format!(
                    "Failed to read extsort generations directory entry at {}: {}",
                    generations_dir.display(),
                    e
                )
            })?;
            if !entry
                .file_type()
                .map_err(|e| {
                    format!(
                        "Failed to inspect extsort generation entry at {}: {}",
                        entry.path().display(),
                        e
                    )
                })?
                .is_dir()
            {
                continue;
            }
            let token = entry.file_name().to_string_lossy().to_string();
            let Some(candidate) = self.complete_generation_candidate(&token)? else {
                continue;
            };
            let replace = match &best {
                None => true,
                Some(current_best) => {
                    candidate.modified > current_best.modified
                        || (candidate.modified == current_best.modified
                            && candidate.token > current_best.token)
                }
            };
            if replace {
                best = Some(candidate);
            }
        }
        Ok(best)
    }

    fn resolve_current_sort_generation_candidate(
        &self,
    ) -> Result<Option<SortGenerationCandidate>, String> {
        let active_candidate = match self.read_active_sort_generation_token()? {
            Some(token) => self.complete_generation_candidate(&token)?,
            None => None,
        };
        let latest_candidate = self.scan_latest_complete_generation_candidate()?;
        Ok(match (active_candidate, latest_candidate) {
            (Some(active), Some(latest))
                if latest.modified > active.modified
                    || (latest.modified == active.modified && latest.token > active.token) =>
            {
                Some(latest)
            }
            (Some(active), _) => Some(active),
            (None, Some(latest)) => Some(latest),
            (None, None) => None,
        })
    }

    fn sync_directory(&self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| {
            format!(
                "Failed to open directory {} for syncing: {}",
                path.display(),
                e
            )
        })?;
        file.sync_all()
            .map_err(|e| format!("Failed to sync directory {}: {}", path.display(), e))
    }

    fn cleanup_sort_artifacts(&self, keep_token: Option<&str>) -> Result<(), String> {
        let generations_dir = self.sort_generations_dir();
        if generations_dir.exists() {
            for entry in fs::read_dir(&generations_dir).map_err(|e| {
                format!(
                    "Failed to read extsort generations directory at {}: {}",
                    generations_dir.display(),
                    e
                )
            })? {
                let entry = entry.map_err(|e| {
                    format!(
                        "Failed to read extsort generations directory entry at {}: {}",
                        generations_dir.display(),
                        e
                    )
                })?;
                if !entry
                    .file_type()
                    .map_err(|e| {
                        format!(
                            "Failed to inspect extsort generation entry at {}: {}",
                            entry.path().display(),
                            e
                        )
                    })?
                    .is_dir()
                {
                    continue;
                }
                let token = entry.file_name().to_string_lossy().to_string();
                if keep_token.is_some_and(|keep| keep == token) {
                    continue;
                }
                if let Err(err) = self.remove_sort_generation(&entry.path()) {
                    log_warning(format!(
                        "Failed to remove obsolete extsort generation {} at {}: {}",
                        token,
                        entry.path().display(),
                        err
                    ));
                }
            }
            if let Err(err) = self.sync_directory(&generations_dir) {
                log_warning(err);
            }
        }

        if keep_token.is_some() {
            for legacy_path in [
                self.legacy_sorted_actions_path(),
                self.legacy_index_path(),
                self.legacy_sorted_source_len_path(),
            ] {
                if let Err(err) = fs::remove_file(&legacy_path) {
                    if err.kind() != std::io::ErrorKind::NotFound {
                        log_warning(format!(
                            "Failed to remove legacy extsort artifact at {}: {}",
                            legacy_path.display(),
                            err
                        ));
                    }
                }
            }
        }

        if let Err(err) = self.sync_directory(&self.storage_dir) {
            log_warning(err);
        }
        Ok(())
    }

    fn repair_sort_generation_state(&self) -> Result<(), String> {
        if let Some(candidate) = self.resolve_current_sort_generation_candidate()? {
            let current_active = self.read_active_sort_generation_token()?;
            if current_active.as_deref() != Some(candidate.token.as_str()) {
                let temp_marker_path = self.active_sort_generation_temp_path(&candidate.token);
                self.write_sort_generation_token_to_path(&temp_marker_path, &candidate.token)?;
                self.publish_sort_generation(&temp_marker_path)?;
            }
            self.cleanup_sort_artifacts(Some(candidate.token.as_str()))?;
            return Ok(());
        }

        self.cleanup_sort_artifacts(None)?;
        Ok(())
    }

    fn sort_generations_dir(&self) -> PathBuf {
        self.storage_dir.join(SORT_GENERATIONS_DIR_NAME)
    }

    fn active_sort_generation_path(&self) -> PathBuf {
        self.storage_dir.join(ACTIVE_SORT_GENERATION_FILE_NAME)
    }

    fn active_sort_generation_temp_path(&self, token: &str) -> PathBuf {
        self.storage_dir
            .join(format!("{SORT_TEMP_FILE_PREFIX}{token}.active_generation"))
    }

    fn sort_generation_dir(&self, token: &str) -> PathBuf {
        self.sort_generations_dir().join(token)
    }

    fn sort_generation_artifact_paths(&self, token: &str) -> ResolvedSortArtifactPaths {
        let generation_dir = self.sort_generation_dir(token);
        ResolvedSortArtifactPaths {
            actions: generation_dir.join("sorted_actions.bin"),
            index: generation_dir.join("sorted_index.bin"),
            source_len: generation_dir.join("sorted_source_len.txt"),
        }
    }

    fn sorted_actions_path_for_generation(&self, token: &str) -> PathBuf {
        self.sort_generation_artifact_paths(token).actions
    }

    fn index_path_for_generation(&self, token: &str) -> PathBuf {
        self.sort_generation_artifact_paths(token).index
    }

    fn sorted_source_len_path_for_generation(&self, token: &str) -> PathBuf {
        self.sort_generation_artifact_paths(token).source_len
    }

    fn legacy_sorted_actions_path(&self) -> PathBuf {
        self.storage_dir.join("sorted_actions.bin")
    }

    fn legacy_index_path(&self) -> PathBuf {
        self.storage_dir.join("sorted_index.bin")
    }

    fn legacy_sorted_source_len_path(&self) -> PathBuf {
        self.storage_dir.join("sorted_source_len.txt")
    }

    fn new_sort_publish_token(&self, source_len: u64) -> String {
        let counter = SORT_TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let timestamp_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);
        format!(
            "{SORT_TEMP_FILE_PREFIX}{}_{}_{}_{}",
            std::process::id(),
            source_len,
            timestamp_millis,
            counter
        )
    }

    fn pending_actions_path(&self) -> PathBuf {
        self.storage_dir.join("pending_actions.bin")
    }

    fn pending_keys_path(&self) -> PathBuf {
        self.storage_dir.join("pending_keys.bin")
    }

    fn sorted_actions_path(&self) -> PathBuf {
        self.legacy_sorted_actions_path()
    }

    fn index_path(&self) -> PathBuf {
        self.legacy_index_path()
    }

    fn sorted_source_len_path(&self) -> PathBuf {
        self.legacy_sorted_source_len_path()
    }
}

#[allow(dead_code)]
fn extsort_storage_dir(base_path: &Path) -> PathBuf {
    let mut storage_dir = base_path.to_path_buf();
    storage_dir.set_extension("extsort");
    storage_dir
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
