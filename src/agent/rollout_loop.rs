use reqwest::Client;

use crate::{
    agent::{
        action_log_schema::TreeActionLogStore, single_dataset::SingleDatasetQuestion,
        state_to_actions::produce_actions_from_state, tree_action_log::TreeActionLog,
        tree_reconstruction::reconstruct_tree,
    },
    llm_model::LlmModelMarker,
    worker_message_tx::log_key_value_pair,
};

pub async fn rollout<M: LlmModelMarker>(
    question: SingleDatasetQuestion,
    rollout_store: TreeActionLogStore,
    llm_callable: M::Callable,
    client: Client,
    rng: &mut impl rand::Rng,
) {
    let mut action_log = rollout_store
        .get(question.id)
        .await
        .unwrap()
        .unwrap_or_else(|| TreeActionLog {
            question: question.clone(),
            actions: vec![],
        });
    assert_eq!(
        action_log.question.id, question.id,
        "TreeActionLog.question.id must match rollout question.id"
    );
    assert_eq!(
        action_log.question.question, question.question,
        "TreeActionLog.question.question must remain immutable"
    );
    assert_eq!(
        action_log.question.final_answer, question.final_answer,
        "TreeActionLog.question.final_answer must remain immutable"
    );

    let mut tree = reconstruct_tree(&action_log);
    log_key_value_pair(
        "status".to_string(),
        format!(
            "Loading {} existing events for question id {}...",
            action_log.actions.len(),
            question.id
        ),
    );

    loop {
        if tree.completed {
            break;
        }
        let new_actions =
            produce_actions_from_state::<M, M::Callable>(&tree, &llm_callable, client.clone(), rng)
                .await;
        assert!(
            !new_actions.is_empty(),
            "produce_actions_from_state must emit at least one action"
        );
        for action in new_actions {
            action_log.actions.push(action);
        }
        rollout_store
            .upsert(question.id, &action_log)
            .await
            .unwrap();
        tree = reconstruct_tree(&action_log);
    }
}
