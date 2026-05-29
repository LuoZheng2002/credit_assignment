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

// for testing, we have many things to report
// we need accuracy with confidence intervals (4 times)

// we also need a histogram of distribution shifting (needs to test 8 times for each question)
// this only applies to one model

// apart from the dataset used in the rollout, we also have different temperature configs, etc.

// For testing, we may be able to do it in one rollout, and extract the accuracy from each dataset respectively.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, clap::ValueEnum)]
pub enum DatasetSplit {
    Training,
    Validation,
    Testing,
}
pub struct AssetFileHybridDataset {
    pub split: DatasetSplit,
}

impl AssetFileHybridDataset {
    pub fn file_path(&self) -> String {
        match self.split {
            DatasetSplit::Training => "datasets/hybrid_train.sqlite".to_string(),
            DatasetSplit::Validation => "datasets/hybrid_val.sqlite".to_string(),
            DatasetSplit::Testing => "datasets/aggregated_test.sqlite".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl AssetFile for AssetFileHybridDataset {
    type FileModel = HybridDatasetStore;
    async fn synchronize(&self) -> research_utility::asset_file::Base64Hash {
        // if the file is stale, we should panic instead of synchronizing, since the file should be updated manually by the user
        hash_file(self.file_path()).unwrap()
    }
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::assume_initialized(self.file_path(), 1).await
    }
}
