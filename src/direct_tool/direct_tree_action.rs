use serde::{Deserialize, Serialize};

use crate::{
    agent::tree::CorrectnessJudgment,
    direct_tool::direct_tree::{SegmentContent, SegmentId},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BranchPosition {
    pub content_index: usize, // must point to a ReasoningOrToolCall content
    pub offset: usize, // the first position that differs from the original trajectory, must be > 0 and < the length of the content tokens
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DirectTreeAction {
    CreateAndFocusTrunkTrajectory {
        content_array: Vec<SegmentContent>,
    },
    CreateAndMoveToBranchPoint {
        target_segment_id: SegmentId, // refers to the end of the segment
        position: BranchPosition,
    },
    MoveToBranchPoint {
        target_segment_id: SegmentId, // refers to the end of the segment
    },
    CreateAndFocusBranchSegment {
        content: Vec<SegmentContent>,
    },
    JudgeFocusedSegmentCorrectness {
        correctness_judgment: CorrectnessJudgment,
    },
}
