use research_utility::worker_message_tx::log_key_value_pair;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::{
    constants::SGLANG_CONTEXT_LENGTH,
    direct_tool::direct_tree::{DirectTree, SegmentContent, SegmentId},
    llm_model::{LlmModelMarker, MyTokenizer, TokenArrayWithLogprob},
    token_array::TokenArray,
    tool_call_python::extract_python_tool_call,
    util::extract_boxed_content,
};

const CONTEXT_LENGTH_SAFETY_MARGIN: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinalAnswer {
    ModelProvided(String),
    Failure(String),
}

impl FinalAnswer {
    pub fn model_answer_text(&self) -> &str {
        match self {
            FinalAnswer::ModelProvided(text) => text,
            FinalAnswer::Failure(text) => text,
        }
    }
}

pub struct DirectTrajectory<M: LlmModelMarker> {
    pub trajectory_contents: Vec<TrajectoryContent<M>>,
    _marker: PhantomData<M>,
}

impl<M: LlmModelMarker> DirectTrajectory<M> {
    pub fn try_get_answer(&self) -> Option<FinalAnswer> {
        let trajectory_length = self
            .trajectory_contents
            .iter()
            .map(|content| content.tokens().len())
            .sum::<usize>();
        if trajectory_length >= SGLANG_CONTEXT_LENGTH - CONTEXT_LENGTH_SAFETY_MARGIN {
            log_key_value_pair(
                "WARNING".to_string(),
                "Trajectory context length exceeded, submitting answer.".to_string(),
            );
            return Some(FinalAnswer::Failure(format!(
                "Context length exceeded (trajectory_length={}, limit={}).",
                trajectory_length, SGLANG_CONTEXT_LENGTH
            )));
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
            final_answer = Some(FinalAnswer::Failure(
                "Model returned only EOS token.".to_string(),
            ));
        }

        if let Some(boxed_content) = extract_boxed_content(&tokens.decode()) {
            final_answer = Some(FinalAnswer::ModelProvided(boxed_content));
        }
        // check for problematic repetition
        if final_answer.is_none() {
            let mut all_tokens: Vec<i32> = vec![];
            for content in &self.trajectory_contents {
                all_tokens.extend_from_slice(content.tokens());
            }
            if has_problematic_repetition(&all_tokens) {
                final_answer = Some(FinalAnswer::Failure(
                    "Generation has problematic repetition.".to_string(),
                ));
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

impl<M: LlmModelMarker> DirectTree<M> {
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

fn has_problematic_repetition(tokens: &[i32]) -> bool {
    let min_subsequence_length = 50; // minimum repeated subsequence length to avoid false positives

    let n = tokens.len();
    if n < min_subsequence_length * 5 {
        return false;
    }

    // Rolling hash over token ids. Verify exact token equality on hash match.
    let base: u64 = 1_000_003;
    let mut pow = vec![0_u64; n + 1];
    let mut prefix = vec![0_u64; n + 1];
    pow[0] = 1;
    for i in 0..n {
        pow[i + 1] = pow[i].wrapping_mul(base);
        let token_as_u64 = (i64::from(tokens[i]) - i64::from(i32::MIN) + 1) as u64;
        prefix[i + 1] = prefix[i].wrapping_mul(base).wrapping_add(token_as_u64);
    }

    let hash = |start: usize, len: usize| -> u64 {
        prefix[start + len].wrapping_sub(prefix[start].wrapping_mul(pow[len]))
    };

    for len in min_subsequence_length..=(n / 5) {
        for start in 0..=(n - 5 * len) {
            let h1 = hash(start, len);
            let h2 = hash(start + len, len);
            let h3 = hash(start + 2 * len, len);
            let h4 = hash(start + 3 * len, len);
            let h5 = hash(start + 4 * len, len);
            if h1 != h2 || h1 != h3 || h1 != h4 || h1 != h5 {
                continue;
            }

            let s1 = &tokens[start..start + len];
            let s2 = &tokens[start + len..start + 2 * len];
            let s3 = &tokens[start + 2 * len..start + 3 * len];
            let s4 = &tokens[start + 3 * len..start + 4 * len];
            let s5 = &tokens[start + 4 * len..start + 5 * len];
            if s1 == s2 && s1 == s3 && s1 == s4 && s1 == s5 {
                return true;
            }
        }
    }

    false
}
