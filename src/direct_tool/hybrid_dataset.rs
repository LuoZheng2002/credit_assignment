use research_utility::{
    asset_file::{AssetFile, hash_file},
    sqlite_store::SqliteStore,
};
use serde::{Deserialize, Serialize};

use crate::json_line_util::HasId;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HybridDatasetQuestion {
    pub flat_id: usize,
    pub dataset_name: String,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
}

impl HasId for HybridDatasetQuestion {
    fn id(&self) -> usize {
        self.flat_id
    }
}

pub type HybridDatasetStore = SqliteStore<usize, HybridDatasetQuestion>;

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
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static
    + PartialEq
    + Eq
    + std::fmt::Debug
{
    const IS_TRAINING: bool;
    fn dataset_file_postfix() -> String;
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct Training;
impl DatasetSplit for Training {
    const IS_TRAINING: bool = true;
    fn dataset_file_postfix() -> String {
        "train".to_string()
    }
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct Validation;
impl DatasetSplit for Validation {
    const IS_TRAINING: bool = false;
    fn dataset_file_postfix() -> String {
        "val".to_string()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct Testing;
impl DatasetSplit for Testing {
    const IS_TRAINING: bool = false;
    fn dataset_file_postfix() -> String {
        "test".to_string()
    }
}

pub struct AssetFileHybridDataset<S: DatasetSplit>(pub std::marker::PhantomData<S>);

impl<S: DatasetSplit> AssetFileHybridDataset<S> {
    pub fn file_path(&self) -> String {
        format!("hybrid_dataset_{}.sqlite", S::dataset_file_postfix())
    }
}

#[async_trait::async_trait]
impl<S: DatasetSplit> AssetFile for AssetFileHybridDataset<S> {
    type FileModel = HybridDatasetStore;
    async fn synchronize(&self) -> research_utility::asset_file::Base64Hash {
        // if the file is stale, we should panic instead of synchronizing, since the file should be updated manually by the user
        hash_file(self.file_path()).unwrap()
    }
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::assume_initialized(self.file_path())
    }
}
