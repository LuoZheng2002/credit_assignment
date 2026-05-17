use crate::{
    direct_tool::{direct_tree::DirectTreeAction, hybrid_dataset_entry::HybridDatasetEntry},
    llm_models::LlmModelMarker,
};

pub async fn rollout<M: LlmModelMarker>(
    question: HybridDatasetEntry,
    loaded_actions: Vec<DirectTreeAction<M>>,
    // llm_callable:
) {
}
