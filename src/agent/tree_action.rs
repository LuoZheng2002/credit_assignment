use serde::{Deserialize, Serialize};

use crate::agent::{trajectory_action::TrajectoryAction, tree::CorrectnessJudgment};

// the following is the signature of each entry in a jsonl log file for reconstructing current tree progress when the program exits abruptly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeAction {
    CreateAndMoveToNode {
        // node_id: usize, // determined by the tree state, should be deterministic based on the number of previous CreateNode calls
        parent_id: Option<usize>,
    },
    // SetCurrentNode {
    //     question_id: usize,
    //     node_id: usize,
    // },
    AddTrajectoryAction {
        action: TrajectoryAction,
    },
    RegisterLeaf {
        node_id: usize,
    },
    JudgeLeafCorrectness {
        node_id: usize,
        correctness_judgment: CorrectnessJudgment,
    },
    ToolWaitViolation,
    TreeComplete,
}
