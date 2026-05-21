use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    agent::tree::CorrectnessJudgment,
    direct_tool::{
        direct_tree_action::DirectTreeAction, direct_tree_action_log::DirectTreeActionLog,
        direct_tree_status::DirectTreeStatus, hybrid_dataset_entry::HybridDatasetQuestion,
    },
    llm_model::{LlmModelMarker, TokenArrayWithLogprob, Top8Candidates},
    token_array::TokenArray,
};

// this tree is similar to the completed tree in src/agent folder, but now it runs on a lightweight tool-calling context instead of a heavy agent framework
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTree<M: LlmModelMarker> {
    pub question: HybridDatasetQuestion,
    // states
    pub status: DirectTreeStatus,
    pub segments: BTreeMap<SegmentId, Segment>, // segment_id -> segment. A segment branched from the middle is destroyed and its id is not reused to avoid hiding sneaky bugs
    pub root_segment_ids: Vec<SegmentId>,
    pub leaf_segment_judgments: BTreeMap<SegmentId, CorrectnessJudgment>,
    pub next_segment_id: usize,
    pub next_segment_temperature: f32,
    pub focused_parent_segment_id: Option<SegmentId>, // the segment after which we create a new branch and rollout until finding the answer
    pub new_branch_start_token: Option<i32>, // the token id for the next branching point, which is determined when we create a branch and will be used in the rollout after branching to determine when to stop and judge the trajectory
    pub completed: bool,
    // hyperparameters
    pub num_trunks: usize,
    pub max_num_total_trajectories: usize,
    pub use_tool: bool,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<M>, // for tokenizer utility
}

pub const NUM_TRUNKS: usize = 4;

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn from_action_log(
        action_log: &DirectTreeActionLog,
        max_num_total_trajectories: usize,
        use_tool: bool,
    ) -> Self {
        let mut tree = Self {
            question: action_log.question.clone(),
            status: DirectTreeStatus::CreatingTrunkTrajectory, // this will be updated when applying actions
            segments: BTreeMap::new(),
            root_segment_ids: vec![],
            leaf_segment_judgments: BTreeMap::new(),
            next_segment_id: 0,
            next_segment_temperature: 1.0,
            focused_parent_segment_id: None,
            new_branch_start_token: None,
            completed: false,
            num_trunks: NUM_TRUNKS, // default value, will not affect the tree structure
            max_num_total_trajectories,
            use_tool,
            _phantom: std::marker::PhantomData::<M>,
        };
        for action in &action_log.actions {
            tree.apply_action(action.clone());
        }
        tree
    }
    // pub fn new(
    //     flat_id: usize,
    //     dataset_name: String,
    //     question_id: usize,
    //     question: String,
    //     correct_answer: String,
    //     num_trunks: usize,
    //     max_num_total_trajectories: usize,
    //     use_tool: bool,
    // ) -> Self {
    //     Self {
    //         flat_id,
    //         dataset_name,
    //         question_id,
    //         question,
    //         correct_answer,
    //         status: DirectTreeStatus::CreatingTrunkTrajectory,
    //         segments: BTreeMap::new(),
    //         root_segment_ids: vec![],
    //         leaf_segment_judgments: BTreeMap::new(),
    //         next_segment_id: 0,
    //         next_segment_temperature: 1.0,
    //         focused_parent_segment_id: None,
    //         new_branch_start_token: None,
    //         completed: false,
    //         num_trunks,
    //         max_num_total_trajectories,
    //         use_tool,
    //         _phantom: std::marker::PhantomData::<M>,
    //     }
    // }
    
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentId(pub usize);

// it has interleaved reasoning and tool response
// we can branch on the reasoning part, but not on the tool response part
// tool response should not be counted towards the segment length
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment {
    pub segment_id: SegmentId,
    pub content: Vec<SegmentContent>,
    pub llm_temperature: f32,
    pub child_ids: Vec<SegmentId>,
    pub parent_id: Option<SegmentId>,
}
// #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
// pub struct ReasoningContentIndex(usize);
// pub struct ReasoningOnlySegmentView<'a> {
//     // pub reasoning_contents: Vec<TokenArrayWithLogprob>,

//     pub corresponding_segment: &'a Segment,
// }

pub struct ReasoningOnlyTokenView<'a> {
    pub flat_index: usize, // the index of the token in the flattened reasoning-only token sequence of the segment
    pub token: i32,
    pub logprobs: Top8Candidates,
    pub content_index_in_segment: usize, // the index of the content in the original segment content array that this token belongs to
    pub token_offset_in_content: usize,  // the offset of the token in the original content tokens
    pub corresponding_segment: &'a Segment,
}

impl Segment {
    pub fn reasoning_only_tokens<'a>(&'a self) -> Vec<ReasoningOnlyTokenView<'a>> {
        let mut views = vec![];
        let mut flat_index = 0;
        for (content_index, content) in self.content.iter().enumerate() {
            if let SegmentContent::ReasoningOrToolCall(tokens) = content {
                for (token_offset, (&token, logprobs)) in
                    tokens.tokens.iter().zip(tokens.logprobs.iter()).enumerate()
                {
                    views.push(ReasoningOnlyTokenView {
                        flat_index,
                        token,
                        logprobs: *logprobs,
                        content_index_in_segment: content_index,
                        token_offset_in_content: token_offset,
                        corresponding_segment: self,
                    });
                    flat_index += 1;
                }
            }
        }
        views
    }
    pub fn first_reasoning_token(&self) -> Option<i32> {
        for content in &self.content {
            if let SegmentContent::ReasoningOrToolCall(tokens) = content {
                if let Some(&first_token) = tokens.tokens.first() {
                    return Some(first_token);
                }
            }
        }
        None
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SegmentContent {
    Prompt(TokenArray),
    ReasoningOrToolCall(TokenArrayWithLogprob),
    ToolResponse(TokenArray),
}

// initially we need to finish 4 full trajectory rollouts.
// we can choose which trajectory to first branch on?

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionEntry {
    pub flat_id: usize,
    pub dataset_name: String,
    pub question_id: usize,
    pub action: DirectTreeAction,
}
