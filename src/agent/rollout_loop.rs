use reqwest::Client;
use std::sync::Arc;

use crate::{
    agent::{
        rollout_batch::LogOrTree,
        state_to_actions::produce_actions_from_state,
        tree::Tree,
        tree_action::TreeAction,
        tree_schema::CompletedTree,
    },
    call_llm::LlmEndpoint,
    direct_answer::generate_raw_answers::LlmModel, worker_message_tx::log_key_value_pair,
};

// it will output action logs and final trajectory
// it will also load existing logs
pub async fn rollout(
    question_id: usize,
    question: String,
    reference_answer: String,
    loaded_events: Vec<TreeAction>,
    llm_endpoint: Arc<LlmEndpoint>,
    client: Client,
    model: LlmModel,
    rng: &mut impl rand::Rng,
    log_or_tree_tx: tokio::sync::mpsc::UnboundedSender<LogOrTree>,
) {
    // create a state machine
    let mut tree = Tree::new(question_id, question.clone(), reference_answer.clone());
    log_key_value_pair(
        "status".to_string(),
        format!(
            "Loading {} existing events for question id {}...",
            loaded_events.len(),
            question_id
        ),
    );
    for event in loaded_events {
        tree.apply_action(event);
    }
    loop {
        if tree.completed {
            break;
        }
        let new_actions =
            produce_actions_from_state(&tree, llm_endpoint.clone(), client.clone(), model, rng)
                .await;
        for action in new_actions {
            tree.apply_action(action.clone());
            log_or_tree_tx.send(LogOrTree::Action(action)).unwrap();
        }
    }
    let step_quality_ratio = tree.get_step_quality_ratio();
    let failed_and_aborted_ratio = tree.get_failed_and_aborted_ratio();
    let trajectory_tree = tree.clone();
    let rollout_trajectory = CompletedTree {
        id: question_id,
        question,
        correct_answer: reference_answer,
        step_quality_ratio,
        failed_and_aborted_ratio,
        trajectory: trajectory_tree,
    };
    log_or_tree_tx
        .send(LogOrTree::Tree(rollout_trajectory))
        .unwrap();
}
