use serde::{Deserialize, Serialize};

use crate::{
    parallel_process_jsonl::{HasId, read_json_lines},
    version_tracking::{AssetFile, Base64Hash, hash_file},
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeepMathQuestion {
    pub id: usize,
    pub question: String,
    pub final_answer: String,
}

impl HasId for DeepMathQuestion {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathQuestionReasoning {
    pub id: usize,
    pub reasoning: String,
    pub final_answer: String,
    pub question: String,
}

impl HasId for DeepMathQuestionReasoning {
    fn id(&self) -> usize {
        self.id
    }
}

// legacy non-agent helper function
pub fn get_questions_with_reasoning_path(dataset_name: &str, num_samples: usize) -> String {
    format!(
        "datasets/{}_ordered_{}_reasoning.jsonl",
        dataset_name, num_samples
    )
}

pub struct AssetFileDataset {
    pub dataset: String,
    pub num_samples: usize,
}

impl AssetFileDataset {
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

impl AssetFile for AssetFileDataset {
    type FileModel = Vec<DeepMathQuestion>;

    fn synchronize(&self) -> Base64Hash {
        hash_file(self.file_path()).unwrap()
    }
    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        read_json_lines(self.file_path()).unwrap()
    }
}
