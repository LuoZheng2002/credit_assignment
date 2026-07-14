use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Cursor, Seek, SeekFrom},
    marker::PhantomData,
    path::{Path, PathBuf},
};

use parking_lot::Mutex;
use research_utility::{
    asset_file::{Base64Hash, hash_file},
    sqlite_store::SqliteStoreKey,
    sqlite_table_array_store::SqliteTableArrayKey,
};
use serde::{Deserialize, Serialize};
use serde_jsonlines::BufReadExt;

use crate::utils::run_python_script;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "S: DatasetSplit")]
pub struct HybridDatasetQuestion<S: DatasetSplit> {
    pub flat_id: QuestionFlatId<S>,
    pub dataset_name: String,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuestionFlatId<S: DatasetSplit>(pub usize, pub PhantomData<S>);

impl<S: DatasetSplit> std::fmt::Debug for QuestionFlatId<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<S: DatasetSplit> std::fmt::Display for QuestionFlatId<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<S: DatasetSplit> Serialize for QuestionFlatId<S> {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, S: DatasetSplit> Deserialize<'de> for QuestionFlatId<S> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let value = usize::deserialize(deserializer)?;
        Ok(Self(value, PhantomData))
    }
}

impl<S: DatasetSplit> SqliteStoreKey for QuestionFlatId<S> {
    fn from_key_text(key_text: &str) -> Result<Self, String>
    where
        Self: Sized,
    {
        Ok(QuestionFlatId(usize::from_key_text(key_text)?, PhantomData))
    }

    fn to_key_text(&self) -> String {
        self.0.to_key_text()
    }
}

impl<S: DatasetSplit> SqliteTableArrayKey for QuestionFlatId<S> {
    fn to_table_key_text(&self) -> String {
        self.0.to_table_key_text()
    }

    fn from_table_key_text(table_key_text: &str) -> Result<Self, String>
    where
        Self: Sized,
    {
        Ok(QuestionFlatId(
            usize::from_table_key_text(table_key_text)?,
            PhantomData,
        ))
    }
}

pub struct HybridDatasetStore<S: DatasetSplit> {
    file_path: PathBuf,
    line_offsets: Vec<u64>,
    cache: Mutex<BTreeMap<usize, HybridDatasetQuestion<S>>>,
}

pub struct HybridDatasetIter<S: DatasetSplit> {
    inner: Box<dyn Iterator<Item = Result<HybridDatasetQuestion<S>, std::io::Error>> + Send>,
    index: usize,
}

impl<S: DatasetSplit> Iterator for HybridDatasetIter<S> {
    type Item = Result<(usize, HybridDatasetQuestion<S>), String>;

    fn next(&mut self) -> Option<Self::Item> {
        let question = match self.inner.next()? {
            Ok(q) => q,
            Err(err) => {
                return Some(Err(format!(
                    "Failed to deserialize JSONL record at index {}: {}",
                    self.index, err
                )));
            }
        };
        let idx = self.index;
        self.index += 1;
        Some(Ok((idx, question)))
    }
}

impl<S: DatasetSplit> HybridDatasetStore<S> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path).map_err(|err| {
            format!(
                "Failed to open hybrid dataset JSONL file {}: {}",
                file_path.display(),
                err
            )
        })?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut line_offsets = Vec::new();
        let mut offset = 0u64;

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).map_err(|err| {
                format!(
                    "Failed to read hybrid dataset JSONL file {} while indexing line offsets: {}",
                    file_path.display(),
                    err
                )
            })?;
            if bytes_read == 0 {
                break;
            }
            line_offsets.push(offset);
            offset += bytes_read as u64;
        }

        Ok(Self {
            file_path,
            line_offsets,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn len(&self) -> usize {
        self.line_offsets.len()
    }

    pub fn get_keys(&self) -> Result<Vec<QuestionFlatId<S>>, String> {
        Ok((0..self.line_offsets.len())
            .map(|flat_id| QuestionFlatId(flat_id, PhantomData))
            .collect())
    }

    pub fn iter(&self) -> Result<HybridDatasetIter<S>, String> {
        let file = File::open(&self.file_path).map_err(|err| {
            format!(
                "Failed to open hybrid dataset JSONL file {} for iteration: {}",
                self.file_path.display(),
                err
            )
        })?;
        let reader = BufReader::new(file);
        let inner = reader.json_lines::<HybridDatasetQuestion<S>>();
        Ok(HybridDatasetIter {
            inner: Box::new(inner),
            index: 0,
        })
    }

    pub fn get(&self, key: QuestionFlatId<S>) -> Result<Option<HybridDatasetQuestion<S>>, String> {
        if let Some(question) = self.cache.lock().get(&key.0).cloned() {
            return Ok(Some(question));
        }

        let Some(offset) = self.line_offsets.get(key.0).copied() else {
            return Ok(None);
        };

        let mut file = File::open(&self.file_path).map_err(|err| {
            format!(
                "Failed to open hybrid dataset JSONL file {} for lookup of key {}: {}",
                self.file_path.display(),
                key.0,
                err
            )
        })?;
        file.seek(SeekFrom::Start(offset)).map_err(|err| {
            format!(
                "Failed to seek hybrid dataset JSONL file {} to offset {} for key {}: {}",
                self.file_path.display(),
                offset,
                key.0,
                err
            )
        })?;

        let mut line = String::new();
        let mut reader = BufReader::new(file);
        let bytes_read = reader.read_line(&mut line).map_err(|err| {
            format!(
                "Failed to read line for key {} from hybrid dataset JSONL file {}: {}",
                key.0,
                self.file_path.display(),
                err
            )
        })?;
        if bytes_read == 0 {
            return Err(format!(
                "Reached EOF while loading key {} from hybrid dataset JSONL file {}",
                key.0,
                self.file_path.display()
            ));
        }

        let mut items =
            BufReader::new(Cursor::new(line.into_bytes())).json_lines::<HybridDatasetQuestion<S>>();
        let question = items
            .next()
            .transpose()
            .map_err(|err| {
                format!(
                    "Failed to deserialize key {} from hybrid dataset JSONL file {}: {}",
                    key.0,
                    self.file_path.display(),
                    err
                )
            })?
            .ok_or_else(|| {
                format!(
                    "Missing JSON value for key {} in hybrid dataset JSONL file {}",
                    key.0,
                    self.file_path.display()
                )
            })?;

        if question.flat_id != key {
            return Err(format!(
                "Hybrid dataset JSONL file {} is out of order: looked up key {}, but deserialized flat_id {}",
                self.file_path.display(),
                key.0,
                question.flat_id.0
            ));
        }

        self.cache.lock().insert(key.0, question.clone());
        Ok(Some(question))
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, clap::ValueEnum)]
pub enum DatasetSplitEnum {
    Training,
    Validation,
    Testing,
}

pub trait DatasetSplit:
    Send
    + Sync
    + Clone
    + Copy
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + std::fmt::Debug
{
    const IS_TRAINING: bool;
    fn dataset_file_postfix() -> String;
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Training;

impl DatasetSplit for Training {
    const IS_TRAINING: bool = true;

    fn dataset_file_postfix() -> String {
        "train".to_string()
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Validation;

impl DatasetSplit for Validation {
    const IS_TRAINING: bool = false;

    fn dataset_file_postfix() -> String {
        "val".to_string()
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Testing;

impl DatasetSplit for Testing {
    const IS_TRAINING: bool = false;

    fn dataset_file_postfix() -> String {
        "test".to_string()
    }
}

pub fn hybrid_dataset_file_path<S: DatasetSplit>() -> String {
    format!("datasets/hybrid_{}.jsonl", S::dataset_file_postfix())
}

/// Ensure the hybrid dataset JSONL file exists. If it's missing, run the
/// corresponding Python download script from `scripts/download_datasets/`.
pub fn ensure_hybrid_dataset_jsonl<S: DatasetSplit>() -> Result<(), String> {
    let file_path = hybrid_dataset_file_path::<S>();
    if Path::new(&file_path).exists() {
        return Ok(());
    }
    let script = format!(
        "scripts/download_datasets/download_hybrid_{}.py",
        S::dataset_file_postfix()
    );
    run_python_script(&script, &[])
}

pub fn hybrid_dataset_hash<S: DatasetSplit>() -> Base64Hash {
    hash_file(hybrid_dataset_file_path::<S>()).unwrap()
}

pub fn open_hybrid_dataset<S: DatasetSplit>() -> HybridDatasetStore<S> {
    // JIT: generate the JSONL file if it doesn't exist yet.
    if let Err(e) = ensure_hybrid_dataset_jsonl::<S>() {
        panic!(
            "Failed to JIT-generate hybrid dataset JSONL at {}: {}",
            hybrid_dataset_file_path::<S>(),
            e
        );
    }

    HybridDatasetStore::open(hybrid_dataset_file_path::<S>()).unwrap_or_else(|e| {
        panic!(
            "Failed to open hybrid dataset JSONL store at {}: {}",
            hybrid_dataset_file_path::<S>(),
            e
        )
    })
}
