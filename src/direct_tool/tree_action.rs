use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        trajectory::FinalAnswer,
        tree::{ContentIndex, SegmentContent, SegmentId},
        tree_spontaneous_branching::TokenPositionInSegment,
    },
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
    AppendSegmentContent(SegmentContent<M>),
    SubmitAnswer(FinalAnswer),
    BranchFromSegmentOrNodeGuided {
        position: TokenPositionInTree, // at least one of content_index and offset must be > 0, indicating the branching happens in the middle of a segment
        new_branch_start_token: i32,
        branch_from_node: bool,
    },
    BranchFromSegmentOrNodeSpontaneous {
        position: TokenPositionInTree,
        branch_from_node: bool,
        position_in_segment: TokenPositionInSegment,
    },
    NoAvailableBranchPoint, // this is in parallel with BranchFromSegment and BranchFromNode, indicating a failure in finding a valid branching token, in which case the agent should conclude the tree
    PrefixTrimNewSegment {
        trim_position: TokenPositionInSegment,
    },
    SplitTreeSegment {
        position: TokenPositionInTree, // must point to a ReasoningOrToolCall content, and the offset must be > 0 and < the length of the content tokens, indicating the splitting happens in the middle of a segment
        branch_from_node: bool, // if true, the branching happens at the boundary between segments, and the position points to the segment boundary
    },
    JudgeAnswer(CorrectnessJudgment),
    AttachSegmentToTree {
        parent_segment_id: SegmentId,
        finalized_content_array: Vec<SegmentContent<M>>,
        correctness_judgment: CorrectnessJudgment,
    },
}

impl<M> Clone for DirectTreeAction<M> {
    fn clone(&self) -> Self {
        match self {
            DirectTreeAction::AppendSegmentContent(content) => {
                DirectTreeAction::AppendSegmentContent(content.clone())
            }
            DirectTreeAction::SubmitAnswer(final_answer) => {
                DirectTreeAction::SubmitAnswer(final_answer.clone())
            }
            DirectTreeAction::BranchFromSegmentOrNodeGuided {
                position,
                new_branch_start_token,
                branch_from_node,
            } => DirectTreeAction::BranchFromSegmentOrNodeGuided {
                position: position.clone(),
                new_branch_start_token: *new_branch_start_token,
                branch_from_node: *branch_from_node,
            },
            DirectTreeAction::BranchFromSegmentOrNodeSpontaneous {
                position,
                branch_from_node,
                position_in_segment,
            } => DirectTreeAction::BranchFromSegmentOrNodeSpontaneous {
                position: position.clone(),
                branch_from_node: *branch_from_node,
                position_in_segment: position_in_segment.clone(),
            },
            DirectTreeAction::NoAvailableBranchPoint => DirectTreeAction::NoAvailableBranchPoint,
            DirectTreeAction::AttachSegmentToTree {
                parent_segment_id,
                finalized_content_array,
                correctness_judgment,
            } => DirectTreeAction::AttachSegmentToTree {
                parent_segment_id: *parent_segment_id,
                finalized_content_array: finalized_content_array.clone(),
                correctness_judgment: correctness_judgment.clone(),
            },
            DirectTreeAction::PrefixTrimNewSegment { trim_position } => {
                DirectTreeAction::PrefixTrimNewSegment {
                    trim_position: trim_position.clone(),
                }
            }
            DirectTreeAction::SplitTreeSegment {
                position,
                branch_from_node,
            } => DirectTreeAction::SplitTreeSegment {
                position: position.clone(),
                branch_from_node: *branch_from_node,
            },
            DirectTreeAction::JudgeAnswer(judgment) => {
                DirectTreeAction::JudgeAnswer(judgment.clone())
            }
        }
    }
}
