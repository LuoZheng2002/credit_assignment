use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};

use arc_swap::ArcSwapOption;
use redb::{
    Database, Key as RedbKey, ReadableTable, TableDefinition, TableError as RedbTableError,
    TypeName, Value as RedbValue, WriteTransaction,
};
use research_utility::progress_text_logger::log_info;

use crate::directories::judgment_cache_path;

const MODEL_ANSWER_JUDGMENT_TABLE_NAME: &str = "model_answer_judgment_cache";

struct ModelAnswerJudgmentCacheSlot {
    store: Arc<Mutex<ModelAnswerJudgmentCacheStore>>,
    model_cli_name: String,
    config_nickname: String,
}

static MODEL_ANSWER_JUDGMENT_CACHE_SLOT: ArcSwapOption<ModelAnswerJudgmentCacheSlot> =
    ArcSwapOption::const_empty();
static MODEL_ANSWER_JUDGMENT_CACHE_INIT_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelAnswerJudgmentCacheKey {
    question_text: String,
    model_answer_string: String,
}

impl ModelAnswerJudgmentCacheKey {
    fn new(question_text: impl Into<String>, model_answer_string: impl Into<String>) -> Self {
        Self {
            question_text: question_text.into(),
            model_answer_string: model_answer_string.into(),
        }
    }

    fn encode_string_pair(question_text: &str, model_answer_string: &str) -> Vec<u8> {
        let question_text_bytes = question_text.as_bytes();
        let model_answer_string_bytes = model_answer_string.as_bytes();
        let mut bytes =
            Vec::with_capacity(16 + question_text_bytes.len() + model_answer_string_bytes.len());
        bytes.extend_from_slice(&(question_text_bytes.len() as u64).to_be_bytes());
        bytes.extend_from_slice(question_text_bytes);
        bytes.extend_from_slice(&(model_answer_string_bytes.len() as u64).to_be_bytes());
        bytes.extend_from_slice(model_answer_string_bytes);
        bytes
    }

    fn decode_string_pair(data: &[u8]) -> (String, String) {
        assert!(data.len() >= 16, "redb cache key must be at least 16 bytes");
        let question_text_len = u64::from_be_bytes(data[..8].try_into().unwrap()) as usize;
        let question_text_start = 8;
        let question_text_end = question_text_start + question_text_len;
        assert!(
            data.len() >= question_text_end + 8,
            "redb cache key is truncated before the model answer length"
        );
        let model_answer_string_len = u64::from_be_bytes(
            data[question_text_end..question_text_end + 8]
                .try_into()
                .unwrap(),
        ) as usize;
        let model_answer_string_start = question_text_end + 8;
        let model_answer_string_end = model_answer_string_start + model_answer_string_len;
        assert!(
            data.len() >= model_answer_string_end,
            "redb cache key is truncated before the model answer bytes"
        );
        let question_text = std::str::from_utf8(&data[question_text_start..question_text_end])
            .expect("redb cache key question text must be valid UTF-8")
            .to_string();
        let model_answer_string =
            std::str::from_utf8(&data[model_answer_string_start..model_answer_string_end])
                .expect("redb cache key model answer must be valid UTF-8")
                .to_string();
        (question_text, model_answer_string)
    }
}

impl RedbValue for ModelAnswerJudgmentCacheKey {
    type SelfType<'a>
        = ModelAnswerJudgmentCacheKey
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
        let (question_text, model_answer_string) = Self::decode_string_pair(data);
        Self {
            question_text,
            model_answer_string,
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        Self::encode_string_pair(&value.question_text, &value.model_answer_string)
    }

    fn type_name() -> TypeName {
        TypeName::new("credit_assignment::ModelAnswerJudgmentCacheKey")
    }
}

impl RedbKey for ModelAnswerJudgmentCacheKey {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        let key1 = Self::decode_string_pair(data1);
        let key2 = Self::decode_string_pair(data2);
        key1.0.cmp(&key2.0).then_with(|| key1.1.cmp(&key2.1))
    }
}

fn cache_table_def() -> TableDefinition<'static, ModelAnswerJudgmentCacheKey, u8> {
    TableDefinition::new(MODEL_ANSWER_JUDGMENT_TABLE_NAME)
}

fn get_or_init_store(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> Arc<Mutex<ModelAnswerJudgmentCacheStore>> {
    // Fast path: already initialized.
    let guard = MODEL_ANSWER_JUDGMENT_CACHE_SLOT.load();
    if let Some(slot) = guard.as_ref() {
        assert_eq!(
            slot.model_cli_name, model_cli_name,
            "Model answer judgment cache was already initialized with model_cli_name={}, \
             but a different model_cli_name={} was requested. \
             Only one (model_cli_name, config_nickname) tuple is supported per program instance.",
            slot.model_cli_name, model_cli_name,
        );
        assert_eq!(
            slot.config_nickname, config_nickname,
            "Model answer judgment cache was already initialized with config_nickname={}, \
             but a different config_nickname={} was requested. \
             Only one (model_cli_name, config_nickname) tuple is supported per program instance.",
            slot.config_nickname, config_nickname,
        );
        return Arc::clone(&slot.store);
    }
    drop(guard);

    // Slow path: create the store and try to install it atomically.
    let _init_guard = MODEL_ANSWER_JUDGMENT_CACHE_INIT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let guard = MODEL_ANSWER_JUDGMENT_CACHE_SLOT.load();
    if let Some(slot) = guard.as_ref() {
        assert_eq!(
            slot.model_cli_name, model_cli_name,
            "Model answer judgment cache was already initialized with model_cli_name={}, \
             but a different model_cli_name={} was requested. \
             Only one (model_cli_name, config_nickname) tuple is supported per program instance.",
            slot.model_cli_name, model_cli_name,
        );
        assert_eq!(
            slot.config_nickname, config_nickname,
            "Model answer judgment cache was already initialized with config_nickname={}, \
             but a different config_nickname={} was requested. \
             Only one (model_cli_name, config_nickname) tuple is supported per program instance.",
            slot.config_nickname, config_nickname,
        );
        return Arc::clone(&slot.store);
    }
    drop(guard);

    let db_path = std::path::PathBuf::from(judgment_cache_path(
        mount_dir,
        model_cli_name,
        config_nickname,
    ));
    let store =
        ModelAnswerJudgmentCacheStore::initialize_if_missing(db_path).unwrap_or_else(|error| {
            panic!("failed to initialize model answer judgment cache: {error}")
        });

    let new_slot = Arc::new(ModelAnswerJudgmentCacheSlot {
        store: Arc::new(Mutex::new(store)),
        model_cli_name: model_cli_name.to_string(),
        config_nickname: config_nickname.to_string(),
    });

    let prev_guard = MODEL_ANSWER_JUDGMENT_CACHE_SLOT.compare_and_swap(
        &None::<Arc<ModelAnswerJudgmentCacheSlot>>,
        Some(Arc::clone(&new_slot)),
    );
    if prev_guard.is_none() {
        // We won the race — our new_slot was installed.
        Arc::clone(&new_slot.store)
    } else {
        // Another thread beat us. Use their store; drop ours.
        let existing = prev_guard.as_ref().unwrap();
        assert_eq!(
            existing.model_cli_name, model_cli_name,
            "Model answer judgment cache was concurrently initialized with model_cli_name={}, \
             but a different model_cli_name={} was requested.",
            existing.model_cli_name, model_cli_name,
        );
        assert_eq!(
            existing.config_nickname, config_nickname,
            "Model answer judgment cache was concurrently initialized with config_nickname={}, \
             but a different config_nickname={} was requested.",
            existing.config_nickname, config_nickname,
        );
        Arc::clone(&existing.store)
    }
}

pub fn get_cached_judgment(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    question_text: impl Into<String>,
    model_answer_string: impl Into<String>,
) -> Result<Option<bool>, String> {
    let store_mutex = get_or_init_store(mount_dir, model_cli_name, config_nickname);
    let mut cache = store_mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.get_cached_judgment(ModelAnswerJudgmentCacheKey::new(
        question_text,
        model_answer_string,
    ))
}

pub fn store_cached_judgment(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    question_text: impl Into<String>,
    model_answer_string: impl Into<String>,
    is_correct: bool,
) -> Result<(), String> {
    let store_mutex = get_or_init_store(mount_dir, model_cli_name, config_nickname);
    let mut cache = store_mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.store_cached_judgment(
        ModelAnswerJudgmentCacheKey::new(question_text, model_answer_string),
        is_correct,
    )
}

pub fn commit_pending_writes_if_any(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<bool, String> {
    let store_mutex = get_or_init_store(mount_dir, model_cli_name, config_nickname);
    let mut cache = store_mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.commit_pending_writes_if_any()
}

struct ModelAnswerJudgmentCacheStore {
    db_path: std::path::PathBuf,
    db: Database,
    write_txn: Option<WriteTransaction>,
    has_pending_writes: bool,
}

impl ModelAnswerJudgmentCacheStore {
    fn initialize_if_missing(db_path: std::path::PathBuf) -> Result<Self, String> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create parent directory for model answer judgment cache at {}: {}",
                    db_path.display(),
                    e
                )
            })?;
        }
        let db = Database::create(&db_path).map_err(|e| {
            format!(
                "Failed to open model answer judgment cache at {}: {}",
                db_path.display(),
                e
            )
        })?;
        let write_txn = Self::begin_write_txn(&db, &db_path)?;
        Ok(Self {
            db_path,
            db,
            write_txn: Some(write_txn),
            has_pending_writes: false,
        })
    }

    fn begin_write_txn(
        db: &Database,
        db_path: &std::path::PathBuf,
    ) -> Result<WriteTransaction, String> {
        db.begin_write().map_err(|e| {
            format!(
                "Failed to begin model answer judgment cache write transaction at {}: {}",
                db_path.display(),
                e
            )
        })
    }

    fn current_write_txn(&self) -> &WriteTransaction {
        self.write_txn
            .as_ref()
            .expect("model answer judgment cache write transaction must exist")
    }

    fn get_cached_judgment(
        &mut self,
        key: ModelAnswerJudgmentCacheKey,
    ) -> Result<Option<bool>, String> {
        let table = match self.current_write_txn().open_table(cache_table_def()) {
            Ok(table) => table,
            Err(RedbTableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => {
                return Err(format!(
                    "Failed to open model answer judgment cache table at {}: {}",
                    self.db_path.display(),
                    e
                ));
            }
        };

        table
            .get(key)
            .map_err(|e| {
                format!(
                    "Failed to fetch model answer judgment cache entry at {}: {}",
                    self.db_path.display(),
                    e
                )
            })
            .map(|maybe_value| maybe_value.map(|value| value.value() != 0))
    }

    fn store_cached_judgment(
        &mut self,
        key: ModelAnswerJudgmentCacheKey,
        is_correct: bool,
    ) -> Result<(), String> {
        let mut table = self
            .write_txn
            .as_ref()
            .expect("model answer judgment cache write transaction must exist")
            .open_table(cache_table_def())
            .map_err(|e| {
                format!(
                    "Failed to open model answer judgment cache table for write at {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
        table.insert(key, u8::from(is_correct)).map_err(|e| {
            format!(
                "Failed to insert model answer judgment cache entry at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        self.has_pending_writes = true;
        Ok(())
    }

    fn commit_pending_writes_if_any(&mut self) -> Result<bool, String> {
        if !self.has_pending_writes {
            return Ok(false);
        }

        log_info("Committing judgment cache...");
        let write_txn = self
            .write_txn
            .take()
            .expect("model answer judgment cache write transaction must exist");
        write_txn.commit().map_err(|e| {
            format!(
                "Failed to commit model answer judgment cache at {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        self.has_pending_writes = false;
        self.write_txn = Some(Self::begin_write_txn(&self.db, &self.db_path)?);
        log_info("Judgment cache committed");
        Ok(true)
    }
}
