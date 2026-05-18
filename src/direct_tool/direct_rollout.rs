use rand::rngs::StdRng;
use reqwest::Client;
use research_utility::worker_message_tx::log_key_value_pair;
use tokio::sync::mpsc;

use crate::{
    direct_tool::{direct_tree::DirectTree, direct_tree_action::DirectTreeAction, hybrid_dataset_entry::HybridDatasetEntry},
    llm_model::LlmModelMarker,
};

pub enum DirectLogOrTree {
    Log(String),
    Action(DirectTreeAction),
}

pub async fn rollout<M: LlmModelMarker>(
    question: HybridDatasetEntry,
    num_trunks: usize,
    max_num_total_trajectories: usize,
    use_tool: bool,
    loaded_actions: Vec<DirectTreeAction>,
    llm_callable: M::Callable,
    client: Client,
    rng: &mut StdRng,
    log_or_tree_tx: mpsc::UnboundedSender<DirectLogOrTree>,
) {
    let mut tree = DirectTree::<M>::new(
        question.flat_id,
        question.dataset_name.clone(),
        question.question_id,
        question.question.clone(),
        question.correct_answer.clone(),
        num_trunks,
        max_num_total_trajectories,
        use_tool,
    );
    log_key_value_pair(
        "status".to_string(),
        format!(
            "Loading {} existing actions for question id {}...",
            loaded_actions.len(),
            question.question_id
        ),
    );
    for action in loaded_actions {
        tree.apply_action(action);
    }
    // loop {
    //     if tree.completed {
    //         break;
    //     }
    //     let new_actions = produce_actions_from_state::<M, M::Callable>(
    //         &tree,
    //         &llm_callable,
    //         client.clone(),
    //         rng,
    //     )
    //     .await;
    //     for action in new_actions {
    //         tree.apply_action(action.clone());
    //         log_or_tree_tx.send(LogOrTree::Action(action)).unwrap();
    //     }
    // }
    // let step_quality_ratio = tree.get_step_quality_ratio();
    // let failed_and_aborted_ratio = tree.get_failed_and_aborted_ratio();
    // let trajectory_tree = tree.clone();
    // let rollout_trajectory = CompletedTree {
    //     id: question_id,
    //     question,
    //     correct_answer: reference_answer,
    //     step_quality_ratio,
    //     failed_and_aborted_ratio,
    //     trajectory: trajectory_tree,
    // };
    // log_or_tree_tx
    //     .send(LogOrTree::Tree(rollout_trajectory))
    //     .unwrap();
}
