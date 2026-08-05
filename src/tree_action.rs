use serde::{Deserialize, Serialize};

use crate::{
    judge_correctness::CorrectnessJudgment,
    trajectory::FinalAnswer,
    tree::{ContentIndex, SegmentContent, SegmentId},
    tree_spontaneous_branching::TokenPositionInSegment,
};

fn default_branch_start_logprob() -> f32 {
    0.0
}

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
        #[serde(default = "default_branch_start_logprob")]
        new_branch_start_logprob: f32,
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
    AttachSegmentToTreeUnjudged {
        parent_segment_id: SegmentId,
        finalized_content_array: Vec<SegmentContent<M>>,
        final_answer: FinalAnswer,
    },
    AttachSegmentToTree {
        parent_segment_id: SegmentId,
        finalized_content_array: Vec<SegmentContent<M>>,
        correctness_judgment: CorrectnessJudgment,
    },
}

#[derive(Deserialize, Debug)]
#[serde(bound(deserialize = ""))]
enum LegacyDirectTreeAction<M> {
    AppendSegmentContent(SegmentContent<M>),
    SubmitAnswer(FinalAnswer),
    BranchFromSegmentOrNodeGuided {
        position: TokenPositionInTree,
        new_branch_start_token: i32,
        branch_from_node: bool,
    },
    BranchFromSegmentOrNodeSpontaneous {
        position: TokenPositionInTree,
        branch_from_node: bool,
        position_in_segment: TokenPositionInSegment,
    },
    NoAvailableBranchPoint,
    PrefixTrimNewSegment {
        trim_position: TokenPositionInSegment,
    },
    SplitTreeSegment {
        position: TokenPositionInTree,
        branch_from_node: bool,
    },
    JudgeAnswer(CorrectnessJudgment),
    AttachSegmentToTree {
        parent_segment_id: SegmentId,
        finalized_content_array: Vec<SegmentContent<M>>,
        correctness_judgment: CorrectnessJudgment,
    },
}

impl<M> From<LegacyDirectTreeAction<M>> for DirectTreeAction<M> {
    fn from(value: LegacyDirectTreeAction<M>) -> Self {
        match value {
            LegacyDirectTreeAction::AppendSegmentContent(content) => {
                Self::AppendSegmentContent(content)
            }
            LegacyDirectTreeAction::SubmitAnswer(final_answer) => Self::SubmitAnswer(final_answer),
            LegacyDirectTreeAction::BranchFromSegmentOrNodeGuided {
                position,
                new_branch_start_token,
                branch_from_node,
            } => Self::BranchFromSegmentOrNodeGuided {
                position,
                new_branch_start_token,
                new_branch_start_logprob: default_branch_start_logprob(),
                branch_from_node,
            },
            LegacyDirectTreeAction::BranchFromSegmentOrNodeSpontaneous {
                position,
                branch_from_node,
                position_in_segment,
            } => Self::BranchFromSegmentOrNodeSpontaneous {
                position,
                branch_from_node,
                position_in_segment,
            },
            LegacyDirectTreeAction::NoAvailableBranchPoint => Self::NoAvailableBranchPoint,
            LegacyDirectTreeAction::PrefixTrimNewSegment { trim_position } => {
                Self::PrefixTrimNewSegment { trim_position }
            }
            LegacyDirectTreeAction::SplitTreeSegment {
                position,
                branch_from_node,
            } => Self::SplitTreeSegment {
                position,
                branch_from_node,
            },
            LegacyDirectTreeAction::JudgeAnswer(judgment) => Self::JudgeAnswer(judgment),
            LegacyDirectTreeAction::AttachSegmentToTree {
                parent_segment_id,
                finalized_content_array,
                correctness_judgment,
            } => Self::AttachSegmentToTree {
                parent_segment_id,
                finalized_content_array,
                correctness_judgment,
            },
        }
    }
}

pub fn deserialize_direct_tree_action_compat<M>(
    payload: &[u8],
) -> Result<DirectTreeAction<M>, bincode::Error> {
    match bincode::deserialize::<DirectTreeAction<M>>(payload) {
        Ok(action) => Ok(action),
        Err(current_error) => match bincode::deserialize::<LegacyDirectTreeAction<M>>(payload) {
            Ok(action) => Ok(action.into()),
            Err(_) => Err(current_error),
        },
    }
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
                new_branch_start_logprob,
                branch_from_node,
            } => DirectTreeAction::BranchFromSegmentOrNodeGuided {
                position: position.clone(),
                new_branch_start_token: *new_branch_start_token,
                new_branch_start_logprob: *new_branch_start_logprob,
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
            DirectTreeAction::AttachSegmentToTreeUnjudged {
                parent_segment_id,
                finalized_content_array,
                final_answer,
            } => DirectTreeAction::AttachSegmentToTreeUnjudged {
                parent_segment_id: *parent_segment_id,
                finalized_content_array: finalized_content_array.clone(),
                final_answer: final_answer.clone(),
            },
        }
    }
}
