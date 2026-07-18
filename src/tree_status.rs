use serde::{Deserialize, Serialize};

use crate::{
    judge_correctness::CorrectnessJudgment,
    llm_model::LlmModelMarker,
    trajectory::FinalAnswer,
    tree::{SegmentContent, SegmentId},
    tree_action::TokenPositionInTree,
    tree_spontaneous_branching::TokenPositionInSegment,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SegmentAttachment {
    Trunk,
    GuidedBranch { parent_segment_id: SegmentId },
    SpontaneousBranch,
}

// for guided branching, the state transition is:
// DeterminingGuidedBranchPoint -> (BranchFromSegment / BranchFromNode) -> SplittingTargetSegment -> (SplitSegment) -> CreatingGuidedBranchingSegment -> (SubmitAnswer) -> JudgingSegment -> (JudgeAnswer) -> AttachingSegmentToTree

// for spontaneous branching, the state transition is:
// CreatingSpontaneousBranchingSegment -> (SubmitAnswer) -> DeterminingSpontaneousBranchingPoint -> (BranchFromSegment / BranchFromNode) -> SplittingTargetSegment -> (SplitSegment) -> AttachingSegmentToTree
#[derive(Serialize, Deserialize, Debug)]
pub enum TrunkSubStatus<M: LlmModelMarker> {
    // next action is AppendSegmentContent or SubmitAnswer
    CollectingSegmentContents {
        cumulative_content_array: Vec<SegmentContent<M>>,
    },
    // next action is JudgeAnswer
    JudgingSegment {
        // needed for next action:
        final_answer: FinalAnswer,
        // needed for future actions:
        finalized_content_array: Vec<SegmentContent<M>>,
    },
    // next action is AttachSegmentToTree
    AttachingToTree {
        correctness_judgment: CorrectnessJudgment,
        finalized_content_array: Vec<SegmentContent<M>>,
        parent_segment_id: SegmentId,
    },
}
impl<M: LlmModelMarker> Clone for TrunkSubStatus<M> {
    fn clone(&self) -> Self {
        match self {
            TrunkSubStatus::CollectingSegmentContents {
                cumulative_content_array,
            } => TrunkSubStatus::CollectingSegmentContents {
                cumulative_content_array: cumulative_content_array.clone(),
            },
            TrunkSubStatus::JudgingSegment {
                final_answer,
                finalized_content_array,
            } => TrunkSubStatus::JudgingSegment {
                final_answer: final_answer.clone(),
                finalized_content_array: finalized_content_array.clone(),
            },
            TrunkSubStatus::AttachingToTree {
                correctness_judgment,
                finalized_content_array,
                parent_segment_id,
            } => TrunkSubStatus::AttachingToTree {
                correctness_judgment: correctness_judgment.clone(),
                finalized_content_array: finalized_content_array.clone(),
                parent_segment_id: *parent_segment_id,
            },
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum GuidedBranchingSubStatus<M: LlmModelMarker> {
    // next action is BranchFromSegment or BranchFromNode or NoAvailableBranchPoint
    DeterminingBranchingPoint,
    // next action is SplitTreeSegment
    SplittingTargetSegment {
        // needed for next action:
        position: TokenPositionInTree,
        branch_from_node: bool,
        // needed for future actions:
        new_branch_start_token: i32,
    },
    // next action is AppendSegmentContent or SubmitAnswer
    CollectingSegmentContents {
        // needed for next action:
        cumulative_content_array: Vec<SegmentContent<M>>,
        // needed for future actions:
        parent_segment_id: SegmentId,
        new_branch_start_token: i32,
    },
    // next action is JudgeAnswer
    JudgingSegment {
        // needed for next action:
        final_answer: FinalAnswer,
        // needed for future actions:
        parent_segment_id: SegmentId,
        finalized_content_array: Vec<SegmentContent<M>>,
    },
    // next action is AttachSegmentToTree
    AttachingToTree {
        // all needed for next action:
        correctness_judgment: CorrectnessJudgment,
        parent_segment_id: SegmentId,
        finalized_content_array: Vec<SegmentContent<M>>,
    },
}

impl<M: LlmModelMarker> Clone for GuidedBranchingSubStatus<M> {
    fn clone(&self) -> Self {
        match self {
            GuidedBranchingSubStatus::DeterminingBranchingPoint => {
                GuidedBranchingSubStatus::DeterminingBranchingPoint
            }
            GuidedBranchingSubStatus::SplittingTargetSegment {
                position,
                branch_from_node,
                new_branch_start_token,
            } => GuidedBranchingSubStatus::SplittingTargetSegment {
                position: position.clone(),
                branch_from_node: *branch_from_node,
                new_branch_start_token: *new_branch_start_token,
            },
            GuidedBranchingSubStatus::CollectingSegmentContents {
                cumulative_content_array,
                parent_segment_id,
                new_branch_start_token,
            } => GuidedBranchingSubStatus::CollectingSegmentContents {
                cumulative_content_array: cumulative_content_array.clone(),
                parent_segment_id: *parent_segment_id,
                new_branch_start_token: *new_branch_start_token,
            },
            GuidedBranchingSubStatus::JudgingSegment {
                final_answer,
                parent_segment_id,
                finalized_content_array,
            } => GuidedBranchingSubStatus::JudgingSegment {
                final_answer: final_answer.clone(),
                parent_segment_id: *parent_segment_id,
                finalized_content_array: finalized_content_array.clone(),
            },
            GuidedBranchingSubStatus::AttachingToTree {
                correctness_judgment,
                parent_segment_id,
                finalized_content_array,
            } => GuidedBranchingSubStatus::AttachingToTree {
                correctness_judgment: correctness_judgment.clone(),
                parent_segment_id: *parent_segment_id,
                finalized_content_array: finalized_content_array.clone(),
            },
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum SpontaneousBranchingSubStatus<M: LlmModelMarker> {
    // next action is AppendSegmentContent or SubmitAnswer
    CollectingSegmentContents {
        cumulative_content_array: Vec<SegmentContent<M>>,
    },
    // next action is BranchFromSegment or BranchFromNode
    DeterminingBranchingPoint {
        // needed for next action:
        finalized_content_array: Vec<SegmentContent<M>>,
        // needed for future actions:
        final_answer: FinalAnswer,
    },
    // next action is PrefixTrimNewSegment
    PrefixTrimmingNewSegment {
        // needed for next action:
        position: TokenPositionInTree,
        position_in_segment: TokenPositionInSegment,
        finalized_content_array: Vec<SegmentContent<M>>,
        // needed for future actions:
        branch_from_node: bool,
        final_answer: FinalAnswer,
    },
    // next action is SplitTreeSegment
    SplittingTargetSegment {
        // needed for next action:
        position: TokenPositionInTree,
        branch_from_node: bool,
        // needed for future actions:
        prefix_trimmed_content_array: Vec<SegmentContent<M>>,
        final_answer: FinalAnswer,
    },
    // next action is JudgeAnswer
    JudgingSegment {
        // needed for next action:
        final_answer: FinalAnswer,
        // needed for future actions:
        parent_segment_id: SegmentId,
        prefix_trimmed_content_array: Vec<SegmentContent<M>>,
    },
    // next action is AttachSegmentToTree
    AttachingToTree {
        // all needed for next action:
        correctness_judgment: CorrectnessJudgment,
        parent_segment_id: SegmentId,
        prefix_trimmed_content_array: Vec<SegmentContent<M>>,
    },
}
impl<M: LlmModelMarker> Clone for SpontaneousBranchingSubStatus<M> {
    fn clone(&self) -> Self {
        match self {
            SpontaneousBranchingSubStatus::CollectingSegmentContents {
                cumulative_content_array,
            } => SpontaneousBranchingSubStatus::CollectingSegmentContents {
                cumulative_content_array: cumulative_content_array.clone(),
            },
            SpontaneousBranchingSubStatus::DeterminingBranchingPoint {
                finalized_content_array,
                final_answer,
            } => SpontaneousBranchingSubStatus::DeterminingBranchingPoint {
                finalized_content_array: finalized_content_array.clone(),
                final_answer: final_answer.clone(),
            },
            SpontaneousBranchingSubStatus::PrefixTrimmingNewSegment {
                position,
                position_in_segment,
                finalized_content_array,
                branch_from_node,
                final_answer,
            } => SpontaneousBranchingSubStatus::PrefixTrimmingNewSegment {
                position: position.clone(),
                position_in_segment: position_in_segment.clone(),
                finalized_content_array: finalized_content_array.clone(),
                branch_from_node: *branch_from_node,
                final_answer: final_answer.clone(),
            },
            SpontaneousBranchingSubStatus::SplittingTargetSegment {
                position,
                branch_from_node,
                prefix_trimmed_content_array,
                final_answer,
            } => SpontaneousBranchingSubStatus::SplittingTargetSegment {
                position: position.clone(),
                branch_from_node: *branch_from_node,
                prefix_trimmed_content_array: prefix_trimmed_content_array.clone(),
                final_answer: final_answer.clone(),
            },
            SpontaneousBranchingSubStatus::JudgingSegment {
                final_answer,
                parent_segment_id,
                prefix_trimmed_content_array,
            } => SpontaneousBranchingSubStatus::JudgingSegment {
                final_answer: final_answer.clone(),
                parent_segment_id: *parent_segment_id,
                prefix_trimmed_content_array: prefix_trimmed_content_array.clone(),
            },
            SpontaneousBranchingSubStatus::AttachingToTree {
                correctness_judgment,
                parent_segment_id,
                prefix_trimmed_content_array,
            } => SpontaneousBranchingSubStatus::AttachingToTree {
                correctness_judgment: correctness_judgment.clone(),
                parent_segment_id: *parent_segment_id,
                prefix_trimmed_content_array: prefix_trimmed_content_array.clone(),
            },
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum DirectTreeStatus<M: LlmModelMarker> {
    WorkingOnTrunk(TrunkSubStatus<M>),
    WorkingOnGuidedBranching(GuidedBranchingSubStatus<M>),
    WorkingOnSpontaneousBranching(SpontaneousBranchingSubStatus<M>),
    Complete,
}

impl<M: LlmModelMarker> Clone for DirectTreeStatus<M> {
    fn clone(&self) -> Self {
        match self {
            DirectTreeStatus::WorkingOnTrunk(sub_status) => {
                DirectTreeStatus::WorkingOnTrunk(sub_status.clone())
            }
            DirectTreeStatus::WorkingOnGuidedBranching(sub_status) => {
                DirectTreeStatus::WorkingOnGuidedBranching(sub_status.clone())
            }
            DirectTreeStatus::WorkingOnSpontaneousBranching(sub_status) => {
                DirectTreeStatus::WorkingOnSpontaneousBranching(sub_status.clone())
            }
            DirectTreeStatus::Complete => DirectTreeStatus::Complete,
        }
    }
}
