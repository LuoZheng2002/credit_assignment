use research_utility::progress_text_logger::{log_info, log_warning};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::atomic::AtomicU64,
    time::{SystemTime, UNIX_EPOCH},
};

use ordered_float::NotNan;

use crate::{
    directories::action_logs_path,
    hybrid_dataset::{DatasetSplit, HybridDatasetQuestion, QuestionFlatId},
    json_toml_utils::write_json,
    llm_model::LlmModelMarker,
    posterior_calculation_config::PosteriorCalculationConfig,
    rollout_config::RolloutConfig,
    tree_action::{DirectTreeAction, deserialize_direct_tree_action_compat},
};

const SORT_GENERATIONS_DIR_NAME: &str = "sorted_generations";
const ACTIVE_SORT_GENERATION_FILE_NAME: &str = "sorted_generation.txt";
const SORT_TEMP_FILE_PREFIX: &str = "sort_tmp_";
const ELAPSED_TIME_FILE_NAME: &str = "elapsed_time.txt";

static SORT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn action_log_config_bundle_path(base_path: &Path) -> PathBuf {
    base_path.with_extension("config_bundle.json")
}

pub fn action_log_config_bundle_file_path(action_logs_path: impl AsRef<Path>) -> PathBuf {
    action_log_config_bundle_path(action_logs_path.as_ref())
}

fn elapsed_time_file_path(base_path: &Path) -> PathBuf {
    base_path.with_extension(ELAPSED_TIME_FILE_NAME)
}

#[derive(Clone)]
pub struct DirectTreeActionLog<M: LlmModelMarker, S: DatasetSplit> {
    pub mount_dir: String,
    pub config_nickname: String,
    pub question: HybridDatasetQuestion<S>,
    pub rollout_config: RolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub use_tool: bool,
    pub fixed_temperature: NotNan<f32>,
    pub actions: Vec<DirectTreeAction<M>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct ActionLogConfigBundle<S: DatasetSplit> {
    pub rollout_config: RolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub use_tool: bool,
    pub fixed_temperature: NotNan<f32>,
}

pub struct ActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    backend: ExtSortActionLogStore<M, S>,
}

impl<M: LlmModelMarker, S: DatasetSplit> ActionLogStore<M, S> {
    pub fn initialize_if_missing(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        ExtSortActionLogStore::<M, S>::initialize_if_missing(db_path)
            .map(|store| Self { backend: store })
    }

    pub fn write_config_bundle_if_missing(
        &self,
        config_bundle: &ActionLogConfigBundle<S>,
    ) -> Result<(), String> {
        self.backend.write_config_bundle_if_missing(config_bundle)
    }

    pub fn get_keys(&self) -> Result<Vec<QuestionFlatId<S>>, String> {
        self.backend.get_keys()
    }

    pub fn load_action_log(
        &self,
        question_flat_id: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        self.backend.load_action_log(question_flat_id)
    }

    pub fn load_or_init_action_log(
        &self,
        question_flat_id: QuestionFlatId<S>,
    ) -> Result<Vec<DirectTreeAction<M>>, String> {
        self.backend.load_or_init_action_log(question_flat_id)
    }

    pub fn append(
        &self,
        question_flat_id: QuestionFlatId<S>,
        action_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        self.backend.append(question_flat_id, action_index, value)
    }

    pub fn sort(&self) -> Result<(), String> {
        self.backend.sort()
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
        self.backend
            .load_or_init_table_sorted(table_key, initialize_rows)
    }

    pub fn append_at(
        &self,
        table_key: QuestionFlatId<S>,
        row_index: usize,
        value: &DirectTreeAction<M>,
    ) -> Result<(), String> {
        self.append(table_key, row_index, value)
    }

    pub fn read_elapsed_time(&self) -> Result<f32, String> {
        self.backend.read_elapsed_time()
    }

    pub fn write_elapsed_time(&self, elapsed_secs: f32) -> Result<(), String> {
        self.backend.write_elapsed_time(elapsed_secs)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutPurpose {
    Training,
    Evaluation,
    Testing {},
}

fn question_flat_id_to_u64<S: DatasetSplit>(key: QuestionFlatId<S>) -> Result<u64, String> {
    u64::try_from(key.0).map_err(|_| format!("QuestionFlatId {} does not fit into u64", key.0))
}

fn question_flat_id_from_u64<S: DatasetSplit>(value: u64) -> Result<QuestionFlatId<S>, String> {
    let usize_value = usize::try_from(value)
        .map_err(|_| format!("QuestionFlatId {value} does not fit into usize"))?;
    Ok(QuestionFlatId(usize_value, PhantomData))
}

#[allow(dead_code)]
struct ExtSortActionLogStore<M: LlmModelMarker, S: DatasetSplit> {
    base_path: PathBuf,
    storage_dir: PathBuf,
    elapsed_time_file: RefCell<Option<File>>,
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
        deserialize_direct_tree_action_compat(&self.action_payload).map_err(|e| {
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
        let elapsed_time_path = elapsed_time_file_path(&base_path);
        let elapsed_time_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&elapsed_time_path)
            .map_err(|e| {
                format!(
                    "Failed to open elapsed time file at {}: {}",
                    elapsed_time_path.display(),
                    e
                )
            })?;
        let store = Self {
            base_path,
            storage_dir,
            elapsed_time_file: RefCell::new(Some(elapsed_time_file)),
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

    fn elapsed_time_file_path(&self) -> PathBuf {
        elapsed_time_file_path(&self.base_path)
    }

    fn read_elapsed_time(&self) -> Result<f32, String> {
        let path = self.elapsed_time_file_path();
        let mut file = self.elapsed_time_file.borrow_mut();
        let file = file
            .as_mut()
            .ok_or_else(|| format!("Elapsed time file handle not open at {}", path.display()))?;
        let metadata = file.metadata().map_err(|e| {
            format!(
                "Failed to get metadata for elapsed time file at {}: {}",
                path.display(),
                e
            )
        })?;
        if metadata.len() == 0 {
            return Ok(0.0);
        }
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            format!(
                "Failed to seek elapsed time file at {}: {}",
                path.display(),
                e
            )
        })?;
        let mut content = String::new();
        file.read_to_string(&mut content).map_err(|e| {
            format!(
                "Failed to read elapsed time file at {}: {}",
                path.display(),
                e
            )
        })?;
        content.trim().parse::<f32>().map_err(|e| {
            format!(
                "Failed to parse elapsed time from {}: {}",
                path.display(),
                e
            )
        })
    }

    fn write_elapsed_time(&self, elapsed_secs: f32) -> Result<(), String> {
        let path = self.elapsed_time_file_path();
        let mut file = self.elapsed_time_file.borrow_mut();
        let file = file
            .as_mut()
            .ok_or_else(|| format!("Elapsed time file handle not open at {}", path.display()))?;
        let content = format!("{:.6}", elapsed_secs);
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            format!(
                "Failed to seek elapsed time file at {}: {}",
                path.display(),
                e
            )
        })?;
        file.write_all(content.as_bytes())
            .map_err(|e| format!("Failed to write elapsed time to {}: {}", path.display(), e))?;
        file.set_len(content.len() as u64).map_err(|e| {
            format!(
                "Failed to truncate elapsed time file at {}: {}",
                path.display(),
                e
            )
        })?;
        file.flush().map_err(|e| {
            format!(
                "Failed to flush elapsed time file at {}: {}",
                path.display(),
                e
            )
        })?;
        Ok(())
    }
}

#[allow(dead_code)]
fn extsort_storage_dir(base_path: &Path) -> PathBuf {
    let mut storage_dir = base_path.to_path_buf();
    storage_dir.set_extension("extsort");
    storage_dir
}

pub fn action_logs_file_path<M: LlmModelMarker, S: DatasetSplit>(
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    action_logs_path::<S>(mount_dir, M::CLI_NAME, config_nickname, epoch)
        .unwrap_or_else(|err| panic!("Failed to resolve action logs path: {}", err))
}

pub fn open_action_logs<M: LlmModelMarker, S: DatasetSplit>(
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
) -> ActionLogStore<M, S> {
    ActionLogStore::<M, S>::initialize_if_missing(action_logs_file_path::<M, S>(
        mount_dir,
        config_nickname,
        epoch,
    ))
    .unwrap_or_else(|e| {
        panic!(
            "Failed to open direct action log store at {}: {}",
            action_logs_file_path::<M, S>(mount_dir, config_nickname, epoch),
            e
        )
    })
}
