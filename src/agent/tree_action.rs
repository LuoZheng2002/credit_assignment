use serde::{Deserialize, Serialize};

use crate::agent::{trajectory_action::TrajectoryAction, tree::CorrectnessJudgment};

// the following is the signature of each entry in a jsonl log file for reconstructing current tree progress when the program exits abruptly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeAction {
    CreateAndMoveToNode {
        question_id: usize,
        // node_id: usize, // determined by the tree state, should be deterministic based on the number of previous CreateNode calls
        parent_id: Option<usize>,
    },
    // SetCurrentNode {
    //     question_id: usize,
    //     node_id: usize,
    // },
    AddTrajectoryAction {
        question_id: usize,
        action: TrajectoryAction,
    },
    RegisterLeaf {
        question_id: usize,
        node_id: usize,
    },
    JudgeLeafCorrectness {
        question_id: usize,
        node_id: usize,
        correctness_judgment: CorrectnessJudgment,
    },
    ToolWaitViolation {
        question_id: usize,
    },
    TreeComplete {
        question_id: usize,
    },
}

impl TreeAction {
    pub fn question_id(&self) -> usize {
        match self {
            TreeAction::CreateAndMoveToNode { question_id, .. } => *question_id,
            // TreeAction::SetCurrentNode { question_id, .. } => *question_id,
            TreeAction::AddTrajectoryAction { question_id, .. } => *question_id,
            TreeAction::RegisterLeaf { question_id, .. } => *question_id,
            TreeAction::JudgeLeafCorrectness { question_id, .. } => *question_id,
            TreeAction::ToolWaitViolation { question_id } => *question_id,
            TreeAction::TreeComplete { question_id } => *question_id,
        }
    }
}
