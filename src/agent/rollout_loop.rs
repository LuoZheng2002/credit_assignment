use reqwest::Client;

use crate::{
    agent::tree_schema::CompletedTree,
    agent::{state_to_actions::produce_actions_from_state, tree::Tree, tree_action::TreeAction},
    direct_answer::generate_raw_answers::LlmModel,
};

// it will output action logs and final trajectory
// it will also load existing logs
pub async fn rollout(
    question_id: usize,
    question: String,
    reference_answer: String,
    loaded_events: Vec<TreeAction>,
    client: Client,
    model: LlmModel,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
    action_tx: tokio::sync::mpsc::UnboundedSender<TreeAction>,
    trajectory_tx: tokio::sync::mpsc::UnboundedSender<CompletedTree>,
) {
    // create a state machine
    let mut tree = Tree::new(question_id, question.clone(), reference_answer.clone());
    println!(
        "Loading {} existing events for question id {}...",
        loaded_events.len(),
        question_id
    );
    for event in loaded_events {
        tree.apply_action(event);
    }
    loop {
        if tree.completed {
            break;
        }
        let new_actions =
            produce_actions_from_state(&tree, client.clone(), model, take_over_mode_decision, rng)
                .await;
        for action in new_actions {
            tree.apply_action(action.clone());
            action_tx.send(action).unwrap();
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
    trajectory_tx.send(rollout_trajectory).unwrap();
}
