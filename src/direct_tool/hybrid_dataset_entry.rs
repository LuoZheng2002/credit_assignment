use research_utility::sqlite_store::SqliteStore;
use serde::{Deserialize, Serialize};

use crate::json_line_util::HasId;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HybridDatasetEntry {
    pub flat_id: usize,
    pub dataset_name: String,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
}

impl HasId for HybridDatasetEntry {
    fn id(&self) -> usize {
        self.flat_id
    }
}


pub type HybridDatasetStore = SqliteStore<usize, HybridDatasetEntry>;