use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{agent::tree::CorrectnessJudgment, llm_model::LlmModelMarker};

// this tree is similar to the completed tree in src/agent folder, but now it runs on a lightweight tool-calling context instead of a heavy agent framework
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTree<M: LlmModelMarker> {
    pub flat_id: usize, // the same flat id as the one in the hybrid dataset
    pub dataset_name: String,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
    pub segments: Vec<Segment<M>>,
    pub root_segment_ids: Vec<usize>,
    pub leaf_segment_judgments: BTreeMap<usize, CorrectnessJudgment>,
    pub next_segment_id: usize,
    // hyperparameters
    pub num_trunks: usize,
    pub max_num_total_trajectories: usize,
    pub use_tool: bool,
}

// it has interleaved reasoning and tool response
// we can branch on the reasoning part, but not on the tool response part
// tool response should not be counted towards the segment length
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment<M: LlmModelMarker> {
    pub segment_id: usize,
    pub content: Vec<SegmentContent<M>>,
    pub child_ids: Vec<usize>,
    pub parent_id: Option<usize>,
}



#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SegmentContent<M: LlmModelMarker> {
    Prompt(M::StringOrTokenArray),
    ReasoningOrToolCall(M::StringOrTokenArray),
    ToolResponse(M::StringOrTokenArray),
}

// initially we need to finish 4 full trajectory rollouts.
// we can choose which trajectory to first branch on?

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BranchPosition {
    pub content_index: usize, // must point to a ReasoningOrToolCall content
    pub offset: usize, // the first position that differs from the original trajectory, must be > 0 and < the length of the content tokens
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DirectTreeAction<M: LlmModelMarker> {
    CreateTrunkTrajectory {
        content: Vec<SegmentContent<M>>,
    },
    CreateAndMoveToBranchPoint {
        target_segment_id: usize,
        position: BranchPosition,
    },
    MoveToBranchPoint {
        parent_segment_id: usize,
    },
    AddBranchSegment {
        content: Vec<SegmentContent<M>>,
    },
    JudgeTrajectoryCorrectness {
        segment_id: usize,
        correctness_judgment: CorrectnessJudgment,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionEntry<M: LlmModelMarker> {
    pub flat_id: usize,
    pub dataset_name: String,
    pub question_id: usize,
    pub action: DirectTreeAction<M>,
}
