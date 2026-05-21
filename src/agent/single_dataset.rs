use serde::{Deserialize, Serialize};

use crate::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    json_line_util::{HasId, read_json_lines},
};

// this is legacy and used for heavily agent implementation in src/agent folder
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SingleDatasetQuestion {
    pub id: usize,
    pub question: String,
    pub final_answer: String,
}

impl HasId for SingleDatasetQuestion {
    fn id(&self) -> usize {
        self.id
    }
}

pub struct AssetFileSingleDataset {
    pub dataset: String,
    pub num_samples: usize,
}

impl AssetFileSingleDataset {
    pub fn file_path(&self) -> String {
        format!(
            "datasets/{}_ordered_{}.jsonl",
            self.dataset, self.num_samples
        )
    }

    pub fn version_tracking_path(&self) -> String {
        unreachable!("Dataset file does not have a tracking file.")
    }
}
#[async_trait::async_trait]
impl AssetFile for AssetFileSingleDataset {
    type FileModel = Vec<SingleDatasetQuestion>;

    async fn synchronize(&self) -> Base64Hash {
        hash_file(self.file_path()).unwrap()
    }
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        read_json_lines(self.file_path()).unwrap()
    }
}
