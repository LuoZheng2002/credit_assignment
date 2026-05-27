use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::direct_tree::{ContentIndex, SegmentContent, SegmentId},
    judge_correctness::CorrectnessJudgment,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenPositionInTree {
    pub segment_id: SegmentId,       // refers to the end of the segment
    pub content_index: ContentIndex, // must point to a ReasoningOrToolCall content
    pub offset: usize, // the first position that differs from the original trajectory, must be > 0 and < the length of the content tokens
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "", deserialize = ""))]
pub enum DirectTreeAction<M> {
    CreateAndJudgeTrunkTrajectory {
        content_array: Vec<SegmentContent<M>>,
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
        contents: Vec<SegmentContent<M>>,
        correctness_judgment: CorrectnessJudgment,
    },
    // JudgeFocusedSegmentCorrectness {
    //     correctness_judgment: CorrectnessJudgment,
    // },
}

impl<M> Clone for DirectTreeAction<M> {
    fn clone(&self) -> Self {
        match self {
            DirectTreeAction::CreateAndJudgeTrunkTrajectory {
                content_array,
                correctness_judgment,
            } => DirectTreeAction::CreateAndJudgeTrunkTrajectory {
                content_array: content_array.clone(),
                correctness_judgment: correctness_judgment.clone(),
            },
            DirectTreeAction::BranchFromSegment {
                position,
                new_branch_start_token,
            } => DirectTreeAction::BranchFromSegment {
                position: position.clone(),
                new_branch_start_token: *new_branch_start_token,
            },
            DirectTreeAction::BranchFromNode {
                position,
                new_branch_start_token,
            } => DirectTreeAction::BranchFromNode {
                position: position.clone(),
                new_branch_start_token: *new_branch_start_token,
            },
            DirectTreeAction::NoAvailableBranchPoint => DirectTreeAction::NoAvailableBranchPoint,
            DirectTreeAction::CreateAndJudgeBranchSegment {
                contents,
                correctness_judgment,
            } => DirectTreeAction::CreateAndJudgeBranchSegment {
                contents: contents.clone(),
                correctness_judgment: correctness_judgment.clone(),
            },
        }
    }
}
