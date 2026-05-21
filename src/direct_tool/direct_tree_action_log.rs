use serde::{Deserialize, Serialize};

use crate::direct_tool::{direct_tree_action::DirectTreeAction, hybrid_dataset_entry::HybridDatasetQuestion};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionLog {
    pub question: HybridDatasetQuestion,
    pub actions: Vec<DirectTreeAction>,
}