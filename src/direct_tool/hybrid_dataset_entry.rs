use research_utility::{asset_file::AssetFile, sqlite_store::SqliteStore};
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

// we no longer need to specify the dataset name
pub struct AssetFileHybridDataset;

impl AssetFileHybridDataset {
    pub fn file_path() -> String {
        "datasets/hybrid_train.sqlite".to_string()
    }
}

impl AssetFile for AssetFileHybridDataset {
    type FileModel = HybridDatasetStore;
    fn fetch(&self) -> Self::FileModel {
        todo!()
    }
    fn synchronize(&self) -> research_utility::asset_file::Base64Hash {
        // if the file is stale, we should panic instead of synchronizing, since the file should be updated manually by the user
        todo!()
    }
}
