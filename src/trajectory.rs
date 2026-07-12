use research_utility::progress_tui_logger::log_warning;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use crate::{
    constants::sglang_context_length,
    hybrid_dataset::DatasetSplit,
    tree::{DirectTree, SegmentContent, SegmentId},
    llm_model::{LlmModelMarker, MyTokenizer, TokenArrayWithLogprob},
    token_array::TokenArray,
    tool_call_python::extract_python_tool_call,
    utils::extract_boxed_content,
};

const CONTEXT_LENGTH_SAFETY_MARGIN: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureMode {
    ContextWindowOverflow,
    OnlyEos,
    TooManyTurns,
}

impl FailureMode {
    pub fn label(&self) -> &'static str {
        match self {
            FailureMode::ContextWindowOverflow => "context window overflow",
            FailureMode::OnlyEos => "only eos",
            FailureMode::TooManyTurns => "too many turns",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinalAnswer {
    ModelProvided(String),
    Failure(FailureMode),
}

impl FinalAnswer {
    pub fn model_answer_text(&self) -> &str {
        match self {
            FinalAnswer::ModelProvided(text) => text,
            FinalAnswer::Failure(mode) => mode.label(),
        }
    }
}

pub struct DirectTrajectory<M: LlmModelMarker> {
    pub trajectory_contents: Vec<TrajectoryContent<M>>,
    _marker: PhantomData<M>,
}

impl<M: LlmModelMarker> DirectTrajectory<M> {
    pub fn to_prompt_tokens(&self) -> Vec<i32> {
        let mut prompt_tokens: Vec<i32> = Vec::new();
        for content in &self.trajectory_contents {
            prompt_tokens.extend_from_slice(content.tokens());
        }
        prompt_tokens
    }
    pub fn try_get_answer(&self, use_tool: bool) -> Option<FinalAnswer> {
        let trajectory_length = self
            .trajectory_contents
            .iter()
            .map(|content| content.tokens().len())
            .sum::<usize>();
        let context_length = sglang_context_length(use_tool);
        if trajectory_length >= context_length - CONTEXT_LENGTH_SAFETY_MARGIN {
            log_warning(format!(
                "Trajectory context length exceeded, submitting answer. trajectory_length={}, limit={}",
                trajectory_length, context_length
            ));
            return Some(FinalAnswer::Failure(FailureMode::ContextWindowOverflow));
        }

        let last_content = self
            .trajectory_contents
            .last()
            .expect("Trajectory needs at least one content");
        let TrajectoryContent::ReasoningOrToolCallComplete(tokens) = last_content else {
            return None;
        };

        let mut final_answer: Option<FinalAnswer> = None;
        let eos_token_id = <M::Tokenizer as MyTokenizer<M>>::eos_token_id();
        if tokens.tokens.len() == 1 && tokens.tokens[0] == eos_token_id {
            final_answer = Some(FinalAnswer::Failure(FailureMode::OnlyEos));
        }

        if let Some(boxed_content) = extract_boxed_content(&tokens.decode()) {
            final_answer = Some(FinalAnswer::ModelProvided(boxed_content));
        }

        if final_answer.is_none() {
            let number_of_turn = self.trajectory_contents.len();
            let limit = 20;
            if number_of_turn > limit {
                final_answer = Some(FinalAnswer::Failure(FailureMode::TooManyTurns));
            }
        }
        final_answer
    }
    pub fn try_get_last_content_tool_call(&self) -> Option<String> {
        let last_content = self
            .trajectory_contents
            .last()
            .expect("Trajectory needs at least one content");
        let TrajectoryContent::ReasoningOrToolCallComplete(tokens) = last_content else {
            return None;
        };
        extract_python_tool_call(tokens.decode())
    }
    pub fn to_decoded_string(&self) -> String {
        self.trajectory_contents
            .iter()
            .map(|content| content.decoded_string())
            .collect::<Vec<String>>()
            .join("    ")
    }
}

#[derive(Debug, Clone)]
pub enum TrajectoryContent<M: LlmModelMarker> {
    Prompt(TokenArray<M>),
    ReasoningOrToolCallIncomplete(TokenArrayWithLogprob<M>),
    // ReasoningOrToolCallComplete{
    //     tokens: TokenArrayWithLogprob,
    //     answer: Option<FinalAnswer>,
    //     tool_call: Option<String>, // the tool call string if this is a tool call, which can be used for better interpretability and debugging, but should not be used for any logic in the code to avoid sneaky bugs where the tool call string is not correctly recorded
    // },
    ReasoningOrToolCallComplete(TokenArrayWithLogprob<M>),
    ToolResponse(TokenArray<M>),
}

impl<M: LlmModelMarker> TrajectoryContent<M> {
    pub fn tokens(&self) -> &[i32] {
        match self {
            TrajectoryContent::ReasoningOrToolCallIncomplete(tokens) => &tokens.tokens,
            TrajectoryContent::ReasoningOrToolCallComplete(tokens) => &tokens.tokens,
            TrajectoryContent::Prompt(tokens) => &tokens.tokens,
            TrajectoryContent::ToolResponse(tokens) => &tokens.tokens,
        }
    }
    pub fn decoded_string(&self) -> String {
        match self {
            TrajectoryContent::ReasoningOrToolCallIncomplete(tokens) => tokens.decode(),
            TrajectoryContent::ReasoningOrToolCallComplete(tokens) => tokens.decode(),
            TrajectoryContent::Prompt(tokens) => tokens.decode(),
            TrajectoryContent::ToolResponse(tokens) => tokens.decode(),
        }
    }
}

impl<'a, M: LlmModelMarker, S: DatasetSplit> DirectTree<'a, M, S> {
    pub fn get_trajectory_segments_till_id(&self, segment_id: SegmentId) -> Vec<SegmentId> {
        let mut segments: Vec<SegmentId> = Vec::new();
        let mut current_segment_id = Some(segment_id);
        while let Some(segment_id) = current_segment_id {
            segments.push(segment_id);
            let segment = self
                .segments
                .get(&segment_id)
                .expect("Parent segment id must exist in segments");
            current_segment_id = segment.parent_id;
        }
        segments.reverse(); // reverse only the order of segments, but keep the content order within each segment
        segments
    }
    pub fn get_trajectory_length_till_id(&self, segment_id: SegmentId) -> usize {
        let segment_ids = self.get_trajectory_segments_till_id(segment_id);
        let mut total_length = 0;
        for id in segment_ids {
            let segment = self
                .segments
                .get(&id)
                .expect("Segment id must exist in segments");
            total_length += segment.token_length();
        }
        total_length
    }
    pub fn get_trajectory(
        &self,
        segment_id: SegmentId,
        additional_contents: &[SegmentContent<M>],
    ) -> DirectTrajectory<M> {
        let segment_ids = self.get_trajectory_segments_till_id(segment_id);
        let mut contents = segment_ids
            .into_iter()
            .map(|id| {
                let segment = self
                    .segments
                    .get(&id)
                    .expect("Segment id must exist in segments");
                segment.content.clone()
            })
            .flatten()
            .collect::<Vec<SegmentContent<M>>>();
        // let mut flattened_contents: Vec<SegmentContent> = contents.into_iter().flatten().collect();
        contents.extend_from_slice(additional_contents);
        let mut trajectory_contents = vec![];
        let mut unpaired_incomplete_reasoning_or_tool_call: Option<TokenArrayWithLogprob<M>> = None;
        for content in contents.iter() {
            match content {
                SegmentContent::Prompt(token_array) => {
                    assert!(unpaired_incomplete_reasoning_or_tool_call.is_none());
                    trajectory_contents.push(TrajectoryContent::Prompt(token_array.clone()));
                }
                SegmentContent::ToolResponse(token_array) => {
                    assert!(unpaired_incomplete_reasoning_or_tool_call.is_none());
                    trajectory_contents.push(TrajectoryContent::ToolResponse(token_array.clone()));
                }
                SegmentContent::ReasoningOrToolCall { tokens, complete } => {
                    let has_unpaired = unpaired_incomplete_reasoning_or_tool_call.is_some();
                    match (has_unpaired, complete) {
                        (false, false) => {
                            unpaired_incomplete_reasoning_or_tool_call = Some(tokens.clone());
                        }
                        (true, false) => {
                            let Some(unpaired) = &mut unpaired_incomplete_reasoning_or_tool_call
                            else {
                                unreachable!();
                            };
                            // combine
                            let mut new_tokens = unpaired.tokens.clone();
                            new_tokens.extend_from_slice(&tokens.tokens);
                            let mut new_logprobs = unpaired.logprobs.clone();
                            new_logprobs.extend_from_slice(&tokens.logprobs);
                            *unpaired = TokenArrayWithLogprob::from_tokens_and_logprobs(
                                new_tokens,
                                new_logprobs,
                            );
                        }
                        (false, true) => {
                            // push a complete content directly
                            trajectory_contents.push(
                                TrajectoryContent::ReasoningOrToolCallComplete(tokens.clone()),
                            );
                        }
                        (true, true) => {
                            let Some(unpaired) = &mut unpaired_incomplete_reasoning_or_tool_call
                            else {
                                unreachable!();
                            };
                            // combine and push
                            let mut new_tokens = unpaired.tokens.clone();
                            new_tokens.extend_from_slice(&tokens.tokens);
                            let mut new_logprobs = unpaired.logprobs.clone();
                            new_logprobs.extend_from_slice(&tokens.logprobs);
                            trajectory_contents.push(
                                TrajectoryContent::ReasoningOrToolCallComplete(
                                    TokenArrayWithLogprob::from_tokens_and_logprobs(
                                        new_tokens,
                                        new_logprobs,
                                    ),
                                ),
                            );
                            unpaired_incomplete_reasoning_or_tool_call = None;
                        }
                    }
                }
            }
        }
        // if there is still an unpaired incomplete reasoning or tool call, we push it as incomplete
        if let Some(unpaired) = unpaired_incomplete_reasoning_or_tool_call {
            trajectory_contents.push(TrajectoryContent::ReasoningOrToolCallIncomplete(unpaired));
        }
        DirectTrajectory {
            trajectory_contents,
            _marker: PhantomData,
        }
    }
}

// impl DirectTrajectory {
//     pub fn from_current_contents(flattened_contents: &[SegmentContent]) -> DirectTrajectory {

//     }
// }

#[allow(dead_code)]
fn has_problematic_repetition(tokens: &[i32]) -> bool {
    let min_subsequence_length = 50; // minimum repeated subsequence length to avoid false positives

    let n = tokens.len();
    if n < min_subsequence_length + 4 {
        return false;
    }

    // Fixed-size rolling hash over token ids.
    let base: u64 = 1_000_003;
    let window_len = min_subsequence_length;
    let num_windows = n - window_len + 1;

    let mut highest_base_pow = 1_u64;
    for _ in 1..window_len {
        highest_base_pow = highest_base_pow.wrapping_mul(base);
    }

    let mut window_hash = 0_u64;
    for &token in &tokens[..window_len] {
        let token_as_u64 = (i64::from(token) - i64::from(i32::MIN) + 1) as u64;
        window_hash = window_hash.wrapping_mul(base).wrapping_add(token_as_u64);
    }

    let mut positions_by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
    positions_by_hash.insert(window_hash, vec![0]);

    for start in 1..num_windows {
        let outgoing = (i64::from(tokens[start - 1]) - i64::from(i32::MIN) + 1) as u64;
        let incoming = (i64::from(tokens[start + window_len - 1]) - i64::from(i32::MIN) + 1) as u64;

        window_hash = window_hash
            .wrapping_sub(outgoing.wrapping_mul(highest_base_pow))
            .wrapping_mul(base)
            .wrapping_add(incoming);
        positions_by_hash
            .entry(window_hash)
            .or_default()
            .push(start);
    }

    for positions in positions_by_hash.values() {
        if positions.len() < 5 {
            continue;
        }

        let position_set: HashSet<usize> = positions.iter().copied().collect();
        let max_position = *positions.last().expect("positions is not empty");
        for i in 0..positions.len() - 4 {
            let first = positions[i];
            for j in i + 1..positions.len() - 3 {
                let stride = positions[j] - first;
                if stride < window_len {
                    continue;
                }
                if first + 4 * stride > max_position {
                    break;
                }

                if position_set.contains(&(first + 2 * stride))
                    && position_set.contains(&(first + 3 * stride))
                    && position_set.contains(&(first + 4 * stride))
                {
                    return true;
                }
            }
        }
    }

    false
}
