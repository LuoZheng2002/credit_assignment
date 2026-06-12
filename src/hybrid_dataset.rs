use research_utility::{
    asset_file::{Base64Hash, hash_file},
    sqlite_store::{SqliteStore, SqliteStoreKey},
    sqlite_table_array_store::SqliteTableArrayKey,
};
use serde::{Deserialize, Serialize};

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
pub struct QuestionFlatId<S: DatasetSplit>(pub usize, pub std::marker::PhantomData<S>);

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
        Ok(Self(value, std::marker::PhantomData))
    }
}

impl<S: DatasetSplit> SqliteStoreKey for QuestionFlatId<S> {
    fn from_key_text(key_text: &str) -> Result<Self, String>
    where
        Self: Sized,
    {
        Ok(QuestionFlatId(
            usize::from_key_text(key_text)?,
            std::marker::PhantomData,
        ))
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
            std::marker::PhantomData,
        ))
    }
}

// pub type HybridDatasetStore = SqliteStore<usize, HybridDatasetQuestion>;

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
    format!("datasets/hybrid_{}.sqlite", S::dataset_file_postfix())
}

pub fn hybrid_dataset_hash<S: DatasetSplit>() -> Base64Hash {
    // If the file is stale, we should panic instead of synchronizing, since the file should be
    // updated manually by the user.
    hash_file(hybrid_dataset_file_path::<S>()).unwrap()
}

pub fn open_hybrid_dataset<S: DatasetSplit>()
-> SqliteStore<QuestionFlatId<S>, HybridDatasetQuestion<S>> {
    SqliteStore::assume_initialized(hybrid_dataset_file_path::<S>(), true).unwrap_or_else(|e| {
        panic!(
            "Failed to open hybrid dataset sqlite store at {}: {}",
            hybrid_dataset_file_path::<S>(),
            e
        )
    })
}
