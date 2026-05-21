use serde::{Deserialize, Serialize};

use crate::{
    agent::tree::CorrectnessJudgment,
    direct_tool::direct_tree::{SegmentContent, SegmentId},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenPositionInTree {
    pub segment_id: SegmentId, // refers to the end of the segment
    pub content_index: usize,  // must point to a ReasoningOrToolCall content
    pub offset: usize, // the first position that differs from the original trajectory, must be > 0 and < the length of the content tokens
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DirectTreeAction {
    CreateAndJudgeTrunkTrajectory {
        content_array: Vec<SegmentContent>,
        correctness_judgment: CorrectnessJudgment,
    },
    BranchFromSegment {
        position: TokenPositionInTree, // at least one of content_index and offset must be > 0, indicating the branching happens in the middle of a segment
        new_branch_start_token: i32,
    },
    BranchFromNode {
        position: TokenPositionInTree, // content index and offset must be both 0, indicating the branching happens at the boundary between segments
        new_branch_start_token: i32,
    },
    NoAvailableBranchPoint, // this is in parallel with BranchFromSegment and BranchFromNode, indicating a failure in finding a valid branching token, in which case the agent should conclude the tree
    CreateAndJudgeBranchSegment {
        contents: Vec<SegmentContent>,
        correctness_judgment: CorrectnessJudgment,
    },
    // JudgeFocusedSegmentCorrectness {
    //     correctness_judgment: CorrectnessJudgment,
    // },
}
