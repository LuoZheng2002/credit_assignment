use rand::rngs::StdRng;
use reqwest::Client;
use research_utility::{sqlite_store::SqliteStore, worker_message_tx::log_key_value_pair};

use crate::{
    direct_tool::{
        direct_tree::DirectTree, direct_tree_action_log::DirectTreeActionLog,
        hybrid_dataset_entry::HybridDatasetQuestion,
    },
    llm_model::LlmModelMarker,
};

pub async fn rollout<M: LlmModelMarker>(
    question: HybridDatasetQuestion,
    rollout_store: SqliteStore<usize, DirectTreeActionLog>,
    max_num_total_trajectories: usize,
    use_tool: bool,
    llm_callable: M::Callable,
    client: Client,
    rng: &mut StdRng,
) {
    let mut action_log = rollout_store
        .get(question.flat_id)
        .await
        .unwrap()
        .unwrap_or_else(|| DirectTreeActionLog {
            question: question.clone(),
            actions: vec![],
        });
    log_key_value_pair(
        "status".to_string(),
        format!(
            "Loading {} existing actions for question id {}...",
            action_log.actions.len(),
            question.question_id
        ),
    );
    loop {
        let tree =
            DirectTree::<M>::from_action_log(&action_log, max_num_total_trajectories, use_tool);
        if tree.completed {
            break;
        }
        let new_actions = tree
            .produce_actions_from_direct_tree(&llm_callable, client.clone(), rng)
            .await;
        for action in new_actions {
            action_log.actions.push(action.clone());
        }
        rollout_store
            .upsert(question.flat_id, &action_log)
            .await
            .unwrap();
    }
    log_key_value_pair("info".to_string(), format!("Rollout {} finished", question.flat_id));
}
