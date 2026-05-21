use crate::agent::{
    tree::Tree, tree_action::TreeAction, tree_action_log::TreeActionLog, tree_schema::CompletedTree,
};

pub fn reconstruct_tree(log: &TreeActionLog) -> Tree {
    let mut tree = Tree::new(
        log.question.id,
        log.question.question.clone(),
        log.question.final_answer.clone(),
    );
    for action in &log.actions {
        tree.apply_action(action.clone());
    }
    tree
}

pub fn reconstruct_completed_tree(log: &TreeActionLog) -> CompletedTree {
    let tree = reconstruct_tree(log);
    CompletedTree {
        id: log.question.id,
        question: log.question.question.clone(),
        correct_answer: log.question.final_answer.clone(),
        step_quality_ratio: tree.get_step_quality_ratio(),
        failed_and_aborted_ratio: tree.get_failed_and_aborted_ratio(),
        trajectory: tree,
    }
}

pub fn is_completed(log: &TreeActionLog) -> bool {
    reconstruct_tree(log).completed
}

pub fn last_action_is_tree_complete(log: &TreeActionLog) -> bool {
    matches!(log.actions.last(), Some(TreeAction::TreeComplete))
}
