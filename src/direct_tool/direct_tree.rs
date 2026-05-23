use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    agent::tree::CorrectnessJudgment,
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree_action::DirectTreeAction,
        direct_tree_action_log::DirectTreeActionLog,
        direct_tree_status::DirectTreeStatus,
        hybrid_dataset::HybridDatasetQuestion,
        posterior_calculation_config::PosteriorCalculationConfig,
        prompt::{prompt_with_tool_call, prompt_without_tool_call},
    },
    llm_model::{LlmModelMarker, TokenArrayWithLogprob, Top8Candidates},
    token_array::TokenArray,
};

use crate::llm_model::MyTokenizer;

// this tree is similar to the completed tree in src/agent folder, but now it runs on a lightweight tool-calling context instead of a heavy agent framework
#[derive(Debug, Clone)]
pub struct DirectTree<M: LlmModelMarker> {
    pub question: HybridDatasetQuestion,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    // states
    pub status: DirectTreeStatus,
    pub segments: BTreeMap<SegmentId, Segment>, // segment_id -> segment. A segment branched from the middle is destroyed and its id is not reused to avoid hiding sneaky bugs
    // pub root_segment_ids: Vec<SegmentId>,
    pub root_segment_id: Option<SegmentId>, // all the trunks share the same root segment, which is the prompt segment
    pub trunk_leaf_segments: Vec<SegmentId>, // the leaf segments of the trunk trajectories
    pub leaf_segment_judgments: BTreeMap<SegmentId, CorrectnessJudgment>,
    pub current_num_trunks: usize,
    pub next_segment_id: usize,
    pub next_segment_temperature: f32,
    pub focused_parent_segment_id: Option<SegmentId>, // the segment after which we create a new branch and rollout until finding the answer
    pub new_branch_start_token: Option<i32>, // the token id for the next branching point, which is determined when we create a branch and will be used in the rollout after branching to determine when to stop and judge the trajectory
    pub completed: bool,
    // hyperparameters
    // pub max_num_trunks: usize,
    // pub max_num_total_trajectories: usize,
    // pub use_tool: bool,
    // #[serde(skip)]
    _phantom: std::marker::PhantomData<M>, // for tokenizer utility
}

// pub const NUM_TRUNKS: usize = 4;

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn from_action_log(
        action_log: &DirectTreeActionLog,
        // max_num_total_trajectories: usize,
        // use_tool: bool,
    ) -> Self {
        let mut tree = Self {
            question: action_log.question.clone(),
            rollout_config: action_log.rollout_config.clone(),
            posterior_calculation_config: action_log.posterior_calculation_config.clone(),
            status: DirectTreeStatus::CreatingTrunkTrajectory, // this will be updated when applying actions
            segments: BTreeMap::new(),
            root_segment_id: None, // all the trunks share the same root segment, which is the prompt segment
            trunk_leaf_segments: Vec::new(), // the leaf segments of the trunk trajectories
            leaf_segment_judgments: BTreeMap::new(),
            current_num_trunks: 0,
            next_segment_id: 0,
            next_segment_temperature: 1.0,
            focused_parent_segment_id: None,
            new_branch_start_token: None,
            completed: false,
            // max_num_trunks: NUM_TRUNKS, // default value, will not affect the tree structure
            // max_num_total_trajectories,
            // use_tool,
            _phantom: std::marker::PhantomData::<M>,
        };
        // we push the prompt segment to the tree before applying any action
        let root_segment_id = SegmentId(tree.next_segment_id);
        tree.next_segment_id += 1;
        let prompt_segment = Self::create_prompt_segment(
            tree.question.question.clone(),
            tree.rollout_config.use_tool,
            root_segment_id,
            tree.next_segment_temperature,
        );
        tree.segments.insert(root_segment_id, prompt_segment);
        tree.root_segment_id = Some(root_segment_id);
        for action in &action_log.actions {
            tree.apply_action(action.clone());
        }
        tree
    }
    fn create_prompt_segment(
        question: String,
        use_tool: bool,
        segment_id: SegmentId,
        temperature: f32,
    ) -> Segment {
        let prompt_string = match use_tool {
            true => prompt_with_tool_call(question),
            false => prompt_without_tool_call(question),
        };
        let tokenized = M::Tokenizer::tokenize(prompt_string);
        Segment {
            segment_id,
            content: vec![SegmentContent::Prompt(tokenized)],
            llm_temperature: temperature,
            child_ids: vec![],
            parent_id: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentId(pub usize);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment {
    pub segment_id: SegmentId,
    pub content: Vec<SegmentContent>,
    pub llm_temperature: f32,
    pub child_ids: Vec<SegmentId>,
    pub parent_id: Option<SegmentId>,
}

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
            if let SegmentContent::ReasoningOrToolCall { tokens, .. } = content {
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
        // for content in &self.content {
        //     if let SegmentContent::ReasoningOrToolCall { tokens, .. } = content {
        //         if let Some(&first_token) = tokens.tokens.first() {
        //             return Some(first_token);
        //         }
        //     }
        // }
        let Some(first_content) = self.content.first() else {
            return None;
        };
        let SegmentContent::ReasoningOrToolCall { tokens, .. } = first_content else {
            panic!(
                "the first content of a segment should be reasoning, but got tool response or prompt."
            );
        };
        tokens.tokens.first().copied()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SegmentContent {
    Prompt(TokenArray),
    ReasoningOrToolCall {
        tokens: TokenArrayWithLogprob,
        complete: bool,
        // answer: Option<FinalAnswer>,
        // tool_call: Option<String>, // the tool call string if this is a tool call, which can be used for better interpretability and debugging, but should not be used for any logic in the code to avoid sneaky bugs where the tool call string is not correctly recorded
    },
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
