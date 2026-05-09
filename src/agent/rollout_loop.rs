use reqwest::Client;

use crate::{
    agent::{
        branching_node_selection::determine_branching_node,
        state_to_actions::produce_actions_from_state,
        trajectory_state::TrajectoryState,
        tree::{Tree, TreeMasterStatus},
        tree_action::TreeAction,
    },
    direct_answer::generate_raw_answers::LlmModel,
    schemas::tree::CompletedTree,
};

pub async fn rollout_loop(
    tree: &mut Tree,
    client: &Client,
    model: LlmModel,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
    action_tx: &tokio::sync::mpsc::UnboundedSender<TreeAction>,
) {
    assert_eq!(
        tree.tree_master_status,
        TreeMasterStatus::WorkingOnTrajectory,
        "produce_working_trajectory requires WorkingOnTrajectory status"
    );
    loop {
        let session_state = TrajectoryState::from_tree(tree);
        println!(
            "[rollout] question index: {}, num actions: {}, num prev steps: {}, num actual steps: {}",
            tree.question_id,
            session_state.total_actions,
            session_state.prev_steps.len(),
            session_state.total_actual_steps
        );

        let new_operations =
            produce_actions_from_state(tree, client.clone(), model, take_over_mode_decision, rng)
                .await;
        drop(session_state);

        for event in new_operations {
            tree.apply_event(event.clone());
            action_tx.send(event).unwrap();
        }
    }
}

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
        tree.apply_event(event);
    }
    loop {
        match tree.tree_master_status {
            TreeMasterStatus::WorkingOnTrajectory => {
                rollout_loop(
                    &mut tree,
                    &client,
                    model,
                    take_over_mode_decision,
                    rng,
                    &action_tx,
                )
                .await;
                tree.tree_master_status = TreeMasterStatus::DeterminingBranchingNode;
            }
            TreeMasterStatus::DeterminingBranchingNode => {
                let should_finalize_rollout = determine_branching_node(&mut tree, rng);
                if should_finalize_rollout {
                    break;
                }
            }
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
