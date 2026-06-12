use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        hybrid_dataset::DatasetSplit,
        prompt::{prompt_with_tool_call, prompt_without_tool_call},
        tree_action::DirectTreeAction,
        tree_action_log::DirectTreeActionLog,
        tree_status::{DirectTreeStatus, TrunkSubStatus},
    },
    judge_correctness::CorrectnessJudgment,
    llm_model::{LlmModelMarker, TokenArrayWithLogprob, Top8Candidates},
    token_array::TokenArray,
};

use crate::llm_model::MyTokenizer;

// this tree is similar to the completed tree in src/agent folder, but now it runs on a lightweight tool-calling context instead of a heavy agent framework
#[derive(Clone)]
pub struct DirectTree<'a, M: LlmModelMarker, S: DatasetSplit> {
    pub action_log: &'a DirectTreeActionLog<M, S>,
    // states
    pub status: DirectTreeStatus<M>,
    pub segments: BTreeMap<SegmentId, Segment<M>>, // segment_id -> segment. A segment branched from the middle is destroyed and its id is not reused to avoid hiding sneaky bugs
    // pub root_segment_ids: Vec<SegmentId>,
    pub root_segment_id: Option<SegmentId>, // all the trunks share the same root segment, which is the prompt segment
    pub trunk_leaf_segments: BTreeSet<SegmentId>, // the leaf segments of the trunk trajectories
    pub leaf_segment_judgments: BTreeMap<SegmentId, CorrectnessJudgment>,
    // pub current_num_trunks: usize,
    pub next_segment_id: usize,
    // pub focused_parent_segment_id: Option<SegmentId>, // the segment after which we create a new branch and rollout until finding the answer
    // pub new_branch_start_token: Option<i32>, // the token id for the next branching point, which is determined when we create a branch and will be used in the rollout after branching to determine when to stop and judge the trajectory
    _phantom: std::marker::PhantomData<M>, // for tokenizer utility
}

// pub const NUM_TRUNKS: usize = 4;

impl<'a, M: LlmModelMarker, S: DatasetSplit> DirectTree<'a, M, S> {
    pub fn from_action_log(action_log: &'a DirectTreeActionLog<M, S>) -> Self {
        let mut tree = Self {
            action_log,
            status: DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::CollectingSegmentContents {
                cumulative_content_array: Vec::new(),
            }), // this will be updated when applying actions
            segments: BTreeMap::new(),
            root_segment_id: None, // all the trunks share the same root segment, which is the prompt segment
            trunk_leaf_segments: BTreeSet::new(), // the leaf segments of the trunk trajectories
            leaf_segment_judgments: BTreeMap::new(),
            // current_num_trunks: 0,
            next_segment_id: 0,
            _phantom: std::marker::PhantomData::<M>,
        };
        // we push the prompt segment to the tree before applying any action
        let root_segment_id = SegmentId(tree.next_segment_id);
        tree.next_segment_id += 1;
        let prompt_segment = Self::create_prompt_segment(
            tree.action_log.question.question.clone(),
            tree.action_log.rollout_config.use_tool,
            root_segment_id,
        );
        tree.segments.insert(root_segment_id, prompt_segment);
        tree.root_segment_id = Some(root_segment_id);
        for action in &tree.action_log.actions {
            tree.apply_action(&action);
        }
        tree
    }
    fn create_prompt_segment(
        question: String,
        use_tool: bool,
        segment_id: SegmentId,
    ) -> Segment<M> {
        let prompt_string = match use_tool {
            true => prompt_with_tool_call(question),
            false => prompt_without_tool_call(question),
        };
        let tokenized = M::Tokenizer::apply_chat_template_and_tokenize(prompt_string, false);
        Segment {
            segment_id,
            content: vec![SegmentContent::Prompt(tokenized)],
            child_ids: vec![],
            parent_id: None,
        }
    }
    pub fn completed(&self) -> bool {
        matches!(self.status, DirectTreeStatus::Complete)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentId(pub usize);

pub type ContentIndex = usize;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment<M> {
    pub segment_id: SegmentId,
    pub content: Vec<SegmentContent<M>>,
    pub child_ids: Vec<SegmentId>,
    pub parent_id: Option<SegmentId>,
}

pub struct ReasoningOnlyTokenView<'a, M> {
    pub flat_index: usize, // the index of the token in the flattened reasoning-only token sequence of the segment
    pub token: i32,
    pub logprobs: Top8Candidates,
    pub content_index_in_segment: ContentIndex, // the index of the content in the original segment content array that this token belongs to
    pub token_offset_in_content: usize, // the offset of the token in the original content tokens
    pub corresponding_segment: &'a Segment<M>,
}

impl<M: LlmModelMarker> Segment<M> {
    pub fn token_length(&self) -> usize {
        let mut total_length = 0;
        for content in &self.content {
            match content {
                SegmentContent::Prompt(tokens) => total_length += tokens.tokens.len(),
                SegmentContent::ReasoningOrToolCall { tokens, .. } => {
                    total_length += tokens.tokens.len()
                }
                SegmentContent::ToolResponse(tokens) => total_length += tokens.tokens.len(),
            }
        }
        total_length
    }
    pub fn reasoning_only_tokens<'a>(&'a self) -> Vec<ReasoningOnlyTokenView<'a, M>> {
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
    pub fn reasoning_only_token_length(&self) -> usize {
        let mut total_length = 0;
        for content in &self.content {
            if let SegmentContent::ReasoningOrToolCall { tokens, .. } = content {
                total_length += tokens.tokens.len();
            }
        }
        total_length
    }
    pub fn first_reasoning_token(&self) -> Option<i32> {
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

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "", deserialize = ""))]
pub enum SegmentContent<M> {
    Prompt(TokenArray<M>),
    ReasoningOrToolCall {
        tokens: TokenArrayWithLogprob<M>,
        complete: bool,
        // answer: Option<FinalAnswer>,
        // tool_call: Option<String>, // the tool call string if this is a tool call, which can be used for better interpretability and debugging, but should not be used for any logic in the code to avoid sneaky bugs where the tool call string is not correctly recorded
    },
    ToolResponse(TokenArray<M>),
}

impl<M> SegmentContent<M> {
    pub fn tokens(&self) -> Vec<i32> {
        match self {
            SegmentContent::Prompt(TokenArray { tokens, .. }) => tokens.clone(),
            SegmentContent::ReasoningOrToolCall { tokens, .. } => tokens.tokens.clone(),
            SegmentContent::ToolResponse(TokenArray { tokens, .. }) => tokens.clone(),
        }
    }
    pub fn tokens_mut(&mut self) -> &mut Vec<i32> {
        match self {
            SegmentContent::Prompt(TokenArray { tokens, .. }) => tokens,
            SegmentContent::ReasoningOrToolCall { tokens, .. } => &mut tokens.tokens,
            SegmentContent::ToolResponse(TokenArray { tokens, .. }) => tokens,
        }
    }
    pub fn trim_prefix(&self, num_tokens_to_trim: usize) -> Option<Self> {
        let mut cloned = self.clone();
        let tokens = cloned.tokens_mut();
        assert!(num_tokens_to_trim <= tokens.len());
        if num_tokens_to_trim == tokens.len() {
            return None;
        }
        tokens.drain(0..num_tokens_to_trim);
        assert!(!tokens.is_empty());
        Some(cloned)
    }
}

impl<M> Clone for SegmentContent<M> {
    fn clone(&self) -> Self {
        match self {
            SegmentContent::Prompt(tokens) => SegmentContent::Prompt(tokens.clone()),
            SegmentContent::ReasoningOrToolCall { tokens, complete } => {
                SegmentContent::ReasoningOrToolCall {
                    tokens: tokens.clone(),
                    complete: *complete,
                }
            }
            SegmentContent::ToolResponse(tokens) => SegmentContent::ToolResponse(tokens.clone()),
        }
    }
}

// initially we need to finish 4 full trajectory rollouts.
// we can choose which trajectory to first branch on?

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct DirectTreeActionEntry<M> {
    pub flat_id: usize,
    pub dataset_name: String,
    pub question_id: usize,
    pub action: DirectTreeAction<M>,
}
