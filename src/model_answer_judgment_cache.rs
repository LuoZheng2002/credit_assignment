use std::{
    cmp::Ordering,
    fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use redb::{
    Database, Key as RedbKey, ReadableTable, TableDefinition, TableError as RedbTableError,
    TypeName, Value as RedbValue, WriteTransaction,
};

const MODEL_ANSWER_JUDGMENT_CACHE_DB_PATH: &str = "cache/model_answer_judgment.redb";
const MODEL_ANSWER_JUDGMENT_TABLE_NAME: &str = "model_answer_judgment_cache";

static MODEL_ANSWER_JUDGMENT_CACHE: OnceLock<Mutex<ModelAnswerJudgmentCacheStore>> =
    OnceLock::new();

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

fn cache_db_path() -> PathBuf {
    PathBuf::from(MODEL_ANSWER_JUDGMENT_CACHE_DB_PATH)
}

fn cache_store() -> &'static Mutex<ModelAnswerJudgmentCacheStore> {
    MODEL_ANSWER_JUDGMENT_CACHE.get_or_init(|| {
        Mutex::new(
            ModelAnswerJudgmentCacheStore::initialize_if_missing(cache_db_path()).unwrap_or_else(
                |error| panic!("failed to initialize model answer judgment cache: {error}"),
            ),
        )
    })
}

pub fn get_cached_judgment(
    question_text: impl Into<String>,
    model_answer_string: impl Into<String>,
) -> Result<Option<bool>, String> {
    let mut cache = cache_store()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.get_cached_judgment(ModelAnswerJudgmentCacheKey::new(
        question_text,
        model_answer_string,
    ))
}

pub fn store_cached_judgment(
    question_text: impl Into<String>,
    model_answer_string: impl Into<String>,
    is_correct: bool,
) -> Result<(), String> {
    let mut cache = cache_store()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.store_cached_judgment(
        ModelAnswerJudgmentCacheKey::new(question_text, model_answer_string),
        is_correct,
    )
}

pub fn commit_pending_writes_if_any() -> Result<bool, String> {
    let mut cache = cache_store()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.commit_pending_writes_if_any()
}

struct ModelAnswerJudgmentCacheStore {
    db_path: PathBuf,
    db: Database,
    write_txn: WriteTransaction,
    has_pending_writes: bool,
}

impl ModelAnswerJudgmentCacheStore {
    fn initialize_if_missing(db_path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
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
            write_txn,
            has_pending_writes: false,
        })
    }

    fn begin_write_txn(db: &Database, db_path: &PathBuf) -> Result<WriteTransaction, String> {
        db.begin_write().map_err(|e| {
            format!(
                "Failed to begin model answer judgment cache write transaction at {}: {}",
                db_path.display(),
                e
            )
        })
    }

    fn current_write_txn(&self) -> &WriteTransaction {
        &self.write_txn
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
        let mut table = self.write_txn.open_table(cache_table_def()).map_err(|e| {
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

        let replacement_txn = Self::begin_write_txn(&self.db, &self.db_path)?;
        let write_txn = std::mem::replace(&mut self.write_txn, replacement_txn);
        let commit_result = write_txn.commit().map_err(|e| {
            format!(
                "Failed to commit model answer judgment cache at {}: {}",
                self.db_path.display(),
                e
            )
        });
        self.has_pending_writes = false;
        match commit_result {
            Ok(()) => Ok(true),
            Err(error) => Err(error),
        }
    }
}
