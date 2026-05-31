use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use reqwest::Client;
use research_utility::log_message::{log_error, log_info, log_key_value_pair, log_warning};

use crate::direct_tool::direct_rollout::StopRequestedError;
use crate::direct_tool::direct_trajectory::{DirectTrajectory, FinalAnswer, TrajectoryContent};
use crate::judge_correctness::{JudgeAnswerModel, judge_final_answer};
use crate::llm_model::MyTokenizer;
use crate::tool_call_python::{PythonToolResponse, PythonToolServerPool, execute_python_tool_call};
use crate::{
    direct_tool::{
        direct_tree::{ContentIndex, DirectTree, SegmentContent, SegmentId},
        direct_tree_action::{DirectTreeAction, TokenPositionInTree},
        direct_tree_posterior::Posterior,
        direct_tree_status::DirectTreeStatus,
    },
    llm_model::{
        LlmCallable, LlmModelMarker, TokenArrayWithLogprob, TokenLogprobCandidate, Top8Candidates,
    },
};

#[derive(Debug, Clone, Copy)]
pub enum BranchingType {
    Node,
    Segment,
}

#[derive(Debug, Clone, Copy)]
pub struct TokenBranchingScore {
    pub token_id: i32,
    pub branching_score: f32,
    pub branching_type: BranchingType,
}

impl<'a, M: LlmModelMarker> DirectTree<'a, M> {
    pub fn calculate_per_token_branching_scores(
        &self,
        segment_uncertainty_scores: &BTreeMap<SegmentId, f32>,
    ) -> BTreeMap<SegmentId, BTreeMap<ContentIndex, BTreeMap<usize, TokenBranchingScore>>> {
        assert!(
            !self.trunk_leaf_segments.is_empty(),
            "Trunk leaf segments must not be empty"
        );
        let trunk_reasoning_only_token_lengths = self
            .trunk_leaf_segments
            .iter()
            .map(|leaf_id| self.trajectory_reasoning_token_length(*leaf_id))
            .collect::<Vec<usize>>();
        let average_trunk_token_length = trunk_reasoning_only_token_lengths.iter().sum::<usize>()
            as f32
            / trunk_reasoning_only_token_lengths.len() as f32;
        assert!(
            average_trunk_token_length > 0.0,
            "Average trunk token length must be greater than zero"
        );
        let mut per_token_branching_scores: BTreeMap<
            SegmentId,
            BTreeMap<ContentIndex, BTreeMap<usize, TokenBranchingScore>>,
        > = BTreeMap::new();

        for (segment_id, segment) in self.segments.iter() {
            let reasoning_only_tokens = segment.reasoning_only_tokens();
            for (token_index_in_reasoning, token_view) in reasoning_only_tokens.iter().enumerate() {
                assert!(token_view.flat_index == token_index_in_reasoning);
                let token_branching_score = if token_index_in_reasoning == 0 {
                    assert!(token_view.content_index_in_segment == 0); // the first token view must correspond to the first content in the segment, which should be a reasoning content
                    let parent_segment = self.segments.get(segment_id).unwrap().parent_id.expect(
                        &format!("A reasoning only segment must have parents since prompt segment is root: {:?}", segment_id),
                    ); // considering that we want the invariant that every reasoning segment has a parent, we need to do special treatment for the first trunk
                    let child_segments = self
                        .segments
                        .get(&parent_segment)
                        .unwrap()
                        .child_ids
                        .clone();
                    let child_tokens = child_segments
                        .iter()
                        .map(|child_segment_id| {
                            self.segments
                                .get(child_segment_id)
                                .expect("Sibling segment id must exist in tree")
                                .first_reasoning_token()
                                .expect("Sibling segment must have at least one reasoning token")
                        })
                        .collect::<Vec<i32>>();

                    let node_uncertainty_score =
                        Self::node_uncertainty_score_from_parent_and_child_ids(
                            parent_segment,
                            &child_segments,
                            segment_uncertainty_scores,
                        );
                    let Some((token_id, best_token_relative_probability)) =
                        Self::best_token_and_relative_probability(
                            &token_view.logprobs, // this might be the cause for the logprobs to be different for gpt models
                            &child_tokens,
                        )
                    else {
                        continue;
                    };
                    let branching_factor_penalty_multiplier =
                        Self::branching_factor_penalty_multiplier(child_segments.len());
                    let branching_score = node_uncertainty_score
                        * best_token_relative_probability
                        * branching_factor_penalty_multiplier;
                    TokenBranchingScore {
                        token_id,
                        branching_score,
                        branching_type: BranchingType::Node,
                    }
                } else {
                    // branching type is segment
                    let segment_uncertainty_score = segment_uncertainty_scores
                        .get(segment_id)
                        .expect("Each segment must have an uncertainty score");
                    let reasoning_only_segment_length = reasoning_only_tokens.len();
                    let first_half_length_after_split = token_index_in_reasoning;
                    let second_half_length_after_split =
                        reasoning_only_segment_length - token_index_in_reasoning;
                    let Some((token_id, best_token_relative_probability)) =
                        Self::best_token_and_relative_probability(
                            &token_view.logprobs,
                            &[token_view.token],
                        )
                    else {
                        continue;
                    };
                    let segment_length_penalty_multiplier = Self::segment_length_penalty_multiplier(
                        first_half_length_after_split,
                        second_half_length_after_split,
                        average_trunk_token_length,
                    );
                    let branching_score = segment_uncertainty_score
                        * best_token_relative_probability
                        * segment_length_penalty_multiplier;
                    TokenBranchingScore {
                        token_id,
                        branching_score,
                        branching_type: BranchingType::Segment,
                    }
                };

                per_token_branching_scores
                    .entry(*segment_id)
                    .or_default()
                    .entry(token_view.content_index_in_segment)
                    .or_default()
                    .insert(token_view.token_offset_in_content, token_branching_score);
            }
        }
        per_token_branching_scores
    }

    pub async fn produce_actions_from_direct_tree(
        &self,
        llm_callable: &M::Callable,
        client: Client,
        python_tool_pool: Arc<PythonToolServerPool>,
        sglang_waiting_workers: Arc<AtomicUsize>,
        stop_signal: Arc<AtomicBool>,
    ) -> Result<Vec<DirectTreeAction<M>>, StopRequestedError> {
        let result = match self.status {
            DirectTreeStatus::CreatingTrunkTrajectory => {
                assert!(self.current_num_trunks < self.action_log.rollout_config.max_num_trunks);
                let root_id = self
                    .root_segment_id
                    .expect("Root segment id must exist when creating trunk trajectory");
                let (content_array, final_answer) = self
                    .generate_continuing_segment_contents(
                        root_id,
                        llm_callable,
                        python_tool_pool.clone(),
                        sglang_waiting_workers.clone(),
                        stop_signal.clone(),
                    )
                    .await?;
                let correctness_judgment = judge_final_answer(
                    &final_answer,
                    &self.action_log.question.correct_answer,
                    &self.action_log.question.question,
                    client,
                    JudgeAnswerModel::DeepseekV4Flash,
                )
                .await;
                vec![DirectTreeAction::CreateAndJudgeTrunkTrajectory {
                    content_array,
                    correctness_judgment,
                }]
            }
            DirectTreeStatus::CreatingOrChoosingBranchPoint => {
                assert!(
                    self.current_num_trunks == self.action_log.rollout_config.max_num_trunks,
                    "Current number of trunks must be equal to the max number of trunks before creating branch point"
                );
                assert!(!self.leaf_segment_judgments.is_empty());
                assert!(
                    self.leaf_segment_judgments.len()
                        < self.action_log.rollout_config.max_num_total_trajectories
                );

                let posteriors = self.calculate_segment_posteriors(None);
                let segment_uncertainty_scores =
                    self.posteriors_to_segment_uncertainty_scores(&posteriors);

                let per_token_branching_scores =
                    self.calculate_per_token_branching_scores(&segment_uncertainty_scores);

                let mut best_token_position: Option<TokenPositionInTree> = None;
                let mut best_token_branching_score: Option<TokenBranchingScore> = None;
                let mut best_branching_score = f32::NEG_INFINITY;

                for (segment_id, content_scores) in per_token_branching_scores.iter() {
                    for (content_index, offset_scores) in content_scores.iter() {
                        for (offset, token_branching_score) in offset_scores.iter() {
                            if token_branching_score.branching_score > best_branching_score {
                                best_branching_score = token_branching_score.branching_score;
                                best_token_position = Some(TokenPositionInTree {
                                    segment_id: *segment_id,
                                    content_index: *content_index,
                                    offset: *offset,
                                });
                                best_token_branching_score = Some(*token_branching_score);
                            }
                        }
                    }
                }
                if let Some(best_token_position) = best_token_position {
                    let best_token_branching_score = best_token_branching_score
                        .expect("Best token score must exist if best token position exists");
                    let action = if matches!(
                        best_token_branching_score.branching_type,
                        BranchingType::Node
                    ) {
                        assert!(best_token_position.content_index == 0);
                        assert!(best_token_position.offset == 0);
                        DirectTreeAction::BranchFromNode {
                            position: best_token_position,
                            new_branch_start_token: best_token_branching_score.token_id,
                        }
                    } else {
                        assert!(
                            best_token_position.content_index > 0 || best_token_position.offset > 0
                        );
                        DirectTreeAction::BranchFromSegment {
                            position: best_token_position,
                            new_branch_start_token: best_token_branching_score.token_id,
                        }
                    };
                    vec![action]
                } else {
                    vec![DirectTreeAction::NoAvailableBranchPoint]
                }
            }
            DirectTreeStatus::CreatingBranchSegment => {
                let focused_parent_segment_id = self
                    .focused_parent_segment_id
                    .expect("Focused parent segment id must exist when creating branch segment");
                let (new_contents, final_answer) = self
                    .generate_continuing_segment_contents(
                        focused_parent_segment_id,
                        llm_callable,
                        python_tool_pool.clone(),
                        sglang_waiting_workers,
                        stop_signal.clone(),
                    )
                    .await?;
                let correctness_judgment = judge_final_answer(
                    &final_answer,
                    &self.action_log.question.correct_answer,
                    &self.action_log.question.question,
                    client,
                    JudgeAnswerModel::DeepseekV4Flash,
                )
                .await;
                let action = DirectTreeAction::CreateAndJudgeBranchSegment {
                    contents: new_contents,
                    correctness_judgment,
                };
                vec![action]
            }
            DirectTreeStatus::SpontaneousBranching => {
                // this is equivalent to determining a branching point and doing a rollout to the end
                let root_id = self
                    .root_segment_id
                    .expect("Root segment id must exist when creating trunk trajectory");
                let (content_array, final_answer) = self
                    .generate_continuing_segment_contents(
                        root_id,
                        llm_callable,
                        python_tool_pool.clone(),
                        sglang_waiting_workers.clone(),
                        stop_signal.clone(),
                    )
                    .await?;
                let correctness_judgment = judge_final_answer(
                    &final_answer,
                    &self.action_log.question.correct_answer,
                    &self.action_log.question.question,
                    client,
                    JudgeAnswerModel::DeepseekV4Flash,
                )
                .await;
                log_info(format!(
                    "Question {}: Created and judged a spontaneous branching, correctness: {}",
                    self.action_log.question.flat_id, correctness_judgment.is_correct
                ));
                let prefix_result = self.find_longest_common_prefix(&content_array);
                let branch_from_existing_node =
                    prefix_result.diverge_position_in_tree.content_index == 0
                        && prefix_result.diverge_position_in_tree.offset == 0;
                let mut remaining_contents: Vec<SegmentContent<M>> = vec![];
                if let Some(first_trimmed) = content_array
                    [prefix_result.diverge_position_in_query.content_index]
                    .trim_prefix(prefix_result.diverge_position_in_query.offset)
                {
                    remaining_contents.push(first_trimmed);
                }
                for content in content_array
                    .into_iter()
                    .skip(prefix_result.diverge_position_in_query.content_index + 1)
                {
                    remaining_contents.push(content);
                }
                assert!(matches!(
                    remaining_contents[0],
                    SegmentContent::ReasoningOrToolCall { .. }
                ));
                let new_branch_start_token = remaining_contents
                    .first()
                    .expect("There should be at least one content in the new trajectory")
                    .tokens()
                    .first()
                    .expect(
                        "The first content in the new trajectory should have at least one token",
                    )
                    .to_owned();
                let branching_action = match branch_from_existing_node {
                    true => DirectTreeAction::BranchFromNode {
                        position: prefix_result.diverge_position_in_tree,
                        new_branch_start_token,
                    },
                    false => DirectTreeAction::BranchFromSegment {
                        position: prefix_result.diverge_position_in_tree,
                        new_branch_start_token,
                    },
                };
                let create_branch_segment_action = DirectTreeAction::CreateAndJudgeBranchSegment {
                    contents: remaining_contents,
                    correctness_judgment,
                };
                vec![branching_action, create_branch_segment_action]
            }
            DirectTreeStatus::Complete => {
                // the tree is complete, no more actions can be taken
                unreachable!()
            }
        };
        Ok(result)
    }
    fn node_uncertainty_score_from_parent_and_children(
        parent_uncertainty_score: f32,
        children_uncertainty_score: &[f32],
    ) -> f32 {
        // parent has the same weight as the sum of children
        let b = children_uncertainty_score.len() as f32;
        (b * parent_uncertainty_score + children_uncertainty_score.iter().sum::<f32>()) / (2.0 * b)
    }
    fn node_uncertainty_score_from_parent_and_child_ids(
        parent_segment_id: SegmentId,
        child_segment_ids: &[SegmentId],
        segment_uncertainty_scores: &BTreeMap<SegmentId, f32>,
    ) -> f32 {
        let parent_uncertainty_score = segment_uncertainty_scores.get(&parent_segment_id).copied();
        let children_uncertainty_score: Vec<f32> = child_segment_ids
            .iter()
            .map(|child_id| {
                segment_uncertainty_scores
                    .get(child_id)
                    .expect("Child segment must have an uncertainty score")
            })
            .cloned()
            .collect();
        if children_uncertainty_score.is_empty() {
            return 0.0;
        }

        if let Some(parent_uncertainty_score) = parent_uncertainty_score {
            return Self::node_uncertainty_score_from_parent_and_children(
                parent_uncertainty_score,
                &children_uncertainty_score,
            );
        }

        children_uncertainty_score.iter().sum::<f32>() / children_uncertainty_score.len() as f32
    }

    fn best_token_and_relative_probability(
        token_logprobs: &Top8Candidates,
        token_ids_to_exclude: &[i32],
    ) -> Option<(i32, f32)> {
        let max_logprob = token_logprobs
            .iter()
            .map(|candidate| candidate.logprob)
            .fold(f32::NEG_INFINITY, f32::max);
        let remaining_token_logprobs: Vec<TokenLogprobCandidate> = token_logprobs
            .iter()
            .filter(|candidate| !token_ids_to_exclude.contains(&candidate.token_id))
            .cloned()
            .collect();
        if remaining_token_logprobs.is_empty() {
            return None;
        }
        let best_candidate = remaining_token_logprobs
            .iter()
            .max_by(|a, b| a.logprob.partial_cmp(&b.logprob).unwrap())
            .unwrap();
        let relative_probability = (best_candidate.logprob - max_logprob).exp();
        if relative_probability < 0.1 {
            // if the best candidate has a relative probability less than 0.1, we do not consider it as a valid branching token candidate, and we skip this token view
            return None;
        }
        Some((best_candidate.token_id, relative_probability))
    }
    pub fn branching_factor_penalty_multiplier(mut branching_factor: usize) -> f32 {
        branching_factor = branching_factor.max(1); // ensure branching factor is at least 1 to avoid zero or negative values
        // assert!(branching_factor >= 1);
        let k_branching_factor = 0.35_f32;
        (-k_branching_factor * (branching_factor as f32 - 1.0)).exp()
    }

    pub fn segment_length_penalty_multiplier(
        first_half_length_after_split: usize,
        second_half_length_after_split: usize,
        average_trunk_token_length: f32,
    ) -> f32 {
        assert!(first_half_length_after_split > 0);
        assert!(second_half_length_after_split > 0);
        let shorter_half_length = std::cmp::min(
            first_half_length_after_split,
            second_half_length_after_split,
        );
        let shorter_half_to_average_ratio =
            (shorter_half_length as f32) / average_trunk_token_length;
        let scaled_ratio = (shorter_half_to_average_ratio * 2.0).clamp(0.0, 1.0);
        // linear relationship
        scaled_ratio
    }
    pub fn posteriors_to_segment_uncertainty_scores(
        &self,
        posteriors: &BTreeMap<SegmentId, Posterior>,
    ) -> BTreeMap<SegmentId, f32> {
        let eps = 1e-8_f32;
        // avoid division by zero
        if posteriors.is_empty() {
            return BTreeMap::new();
        }
        // the raw uncertainty or signal-to-noise ratio score
        let mean_div_stds: BTreeMap<SegmentId, f32> = posteriors
            .iter()
            .map(|(segment_id, posterior)| {
                let std = posterior.log_std.exp();
                (*segment_id, posterior.mean / (std + eps))
            })
            .collect();

        let mean_of_mean_div_stds =
            mean_div_stds.values().sum::<f32>() / mean_div_stds.len() as f32;
        let std_of_mean_div_stds = (mean_div_stds
            .values()
            .map(|value| (value - mean_of_mean_div_stds).powi(2))
            .sum::<f32>()
            / mean_div_stds.len() as f32)
            .sqrt();

        let mean_div_stds_normalized: BTreeMap<SegmentId, f32> = mean_div_stds
            .iter()
            .map(|(segment_id, mean_div_std)| {
                let mean_div_std_norm =
                    (*mean_div_std - mean_of_mean_div_stds) / (std_of_mean_div_stds + eps);
                (*segment_id, mean_div_std_norm)
            })
            .collect();
        // then we need to find uncertainty score that ranges from 0 to 1
        let uncertainty_scores: BTreeMap<SegmentId, f32> = mean_div_stds_normalized
            .iter()
            .map(|(segment_id, mean_div_std_norm)| {
                let alpha = 1.0_f32;
                let uncertainty_score = (-alpha * mean_div_std_norm.powi(2)).exp();
                (*segment_id, uncertainty_score)
            })
            .collect();
        uncertainty_scores
    }
    async fn generate_continuing_segment_contents(
        &self,
        // mut current_contents: Vec<SegmentContent>,
        target_segment_id: SegmentId,
        // client: Client,
        llm_callable: &M::Callable,
        python_tool_pool: Arc<PythonToolServerPool>,
        sglang_waiting_workers: Arc<AtomicUsize>,
        stop_signal: Arc<AtomicBool>,
        // rng: &mut StdRng,
    ) -> Result<(Vec<SegmentContent<M>>, FinalAnswer), StopRequestedError> {
        let mut continuing_contents = Vec::new();
        loop {
            let trajectory = self.get_trajectory(target_segment_id, &continuing_contents);
            if let Some(answer) = trajectory.try_get_answer() {
                if continuing_contents.is_empty() {
                    log_error(format!(
                        "trajectory contents: {}",
                        trajectory.to_decoded_string()
                    ));
                    panic!(
                        "The trajectory should produce some continuing content before producing the answer"
                    );
                }
                return Ok((continuing_contents, answer));
            }
            let next_content = generate_next_segment_content::<M>(
                self.action_log.question.flat_id,
                &trajectory,
                llm_callable,
                python_tool_pool.clone(),
                sglang_waiting_workers.clone(),
                stop_signal.clone(),
            )
            .await?;
            if matches!(next_content, SegmentContent::ReasoningOrToolCall { .. }) {
                if let Some(SegmentContent::ReasoningOrToolCall { complete, .. }) =
                    continuing_contents.last_mut()
                {
                    if *complete {
                        *complete = false;
                    }
                }
            }
            continuing_contents.push(next_content);
        }
    }
    pub fn trajectory_reasoning_token_length(&self, segment_id: SegmentId) -> usize {
        let mut trajectory_segments = Vec::new();
        let mut current_segment = Some(
            self.segments
                .get(&segment_id)
                .expect("Segment id must exist in tree"),
        );
        while let Some(segment) = current_segment {
            trajectory_segments.push(segment);
            if let Some(parent_id) = segment.parent_id {
                current_segment = self.segments.get(&parent_id);
            } else {
                break;
            }
        }
        trajectory_segments
            .iter()
            .map(|segment| segment.reasoning_only_token_length())
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct SegmentCandidate {
    pub segment_id: SegmentId,
    pub position: TokenPositionInTree,
    pub score: f32,
}
#[derive(Debug, Clone)]
pub struct NodeCandidate {
    pub node_id: SegmentId,
    pub score: f32,
}

// pub enum SegmentContentResult {
//     Continue(SegmentContent),
//     Stop(SegmentContent),
// }

// when can a trajectory end?
// 1. found answer in \boxed{}
// 2. context length exceeded
// 3. other scenarios that require termination

fn fallback_top8_for_token(token_id: i32) -> Top8Candidates {
    let mut top8 = [TokenLogprobCandidate {
        token_id,
        logprob: f32::NEG_INFINITY,
    }; 8];
    top8[0] = TokenLogprobCandidate {
        token_id,
        logprob: 0.0,
    };
    top8
}

fn single_eos_response<M: LlmModelMarker>() -> TokenArrayWithLogprob<M> {
    let eos_token_id = <M::Tokenizer as MyTokenizer<M>>::eos_token_id();
    TokenArrayWithLogprob::from_tokens_and_logprobs(
        vec![eos_token_id],
        vec![fallback_top8_for_token(eos_token_id)],
    )
}

fn concise_failure_reason(error: &str) -> String {
    const MAX_REASON_CHARS: usize = 120;
    let first_line = error.lines().next().unwrap_or("").trim();
    let first_line_len = first_line.chars().count();
    if first_line_len <= MAX_REASON_CHARS {
        return first_line.to_string();
    }
    let truncated: String = first_line.chars().take(MAX_REASON_CHARS).collect();
    format!("{}...", truncated)
}

async fn generate_reasoning_or_tool_call_content<M: LlmModelMarker>(
    _question_flat_id: usize,
    // current_content: &[SegmentContent],
    trajectory: &DirectTrajectory<M>,
    llm_callable: &M::Callable,
    sglang_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
) -> Result<SegmentContent<M>, StopRequestedError> {
    let prompt_tokens = trajectory.to_prompt_tokens();
    let mut response = None;
    for trial in 1..=3 {
        let num_waiting_workers = sglang_waiting_workers.fetch_add(1, Ordering::SeqCst) + 1;
        log_key_value_pair("sglang_waiting_workers", num_waiting_workers.to_string());
        if stop_signal.load(Ordering::Relaxed) {
            return Err(StopRequestedError);
        }
        let generation_result = llm_callable
            .generate_tokens_with_logprobs(prompt_tokens.clone(), true, 1.0, true)
            .await;
        let num_waiting_workers = sglang_waiting_workers.fetch_sub(1, Ordering::SeqCst) - 1;
        log_key_value_pair(
            "sglang_waiting_workers".to_string(),
            num_waiting_workers.to_string(),
        );
        match generation_result {
            Ok(result) => {
                response = Some(result);
                break;
            }
            Err(error) => {
                log_warning(format!(
                    "generate_tokens_with_logprobs failed on trial {}/3: {}",
                    trial, error
                ));
            }
        }
    }

    let response = response.unwrap_or_else(|| single_eos_response::<M>());
    let decoded_response = response.decode();
    if response.tokens.is_empty() {
        panic!(
            "LLM returned empty response. Decoded string: '{}', tokens: {:?}",
            decoded_response, response.tokens
        );
    }
    let result = SegmentContent::ReasoningOrToolCall {
        tokens: response,
        complete: true,
    };
    Ok(result)
}

async fn generate_next_segment_content<M: LlmModelMarker>(
    question_flat_id: usize,
    trajectory: &DirectTrajectory<M>,
    // current_content: &[SegmentContent],
    // client: Client,
    llm_callable: &M::Callable,
    python_tool_pool: Arc<PythonToolServerPool>,
    sglang_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
    // rng: &mut StdRng,
) -> Result<SegmentContent<M>, StopRequestedError> {
    let last_trajectory_content = trajectory
        .trajectory_contents
        .last()
        .expect("Current content must not be empty");
    match last_trajectory_content {
        TrajectoryContent::Prompt(_)
        | TrajectoryContent::ToolResponse(_)
        | TrajectoryContent::ReasoningOrToolCallIncomplete(_) => {
            let new_content = generate_reasoning_or_tool_call_content::<M>(
                question_flat_id,
                trajectory,
                llm_callable,
                sglang_waiting_workers,
                stop_signal,
            )
            .await?;
            Ok(new_content)
        }
        TrajectoryContent::ReasoningOrToolCallComplete(_) => {
            if let Some(tool_call) = trajectory.try_get_last_content_tool_call() {
                // log_key_value_pair("info".to_string(), "Executing a tool call".to_string());
                let tool_response = execute_python_tool_call(&python_tool_pool, &tool_call).await;
                match &tool_response {
                    PythonToolResponse::PythonSuccess(_) => {}
                    PythonToolResponse::PythonError(error) => {
                        log_warning(format!(
                            "Tool call failed. flat_id={} reason={}",
                            question_flat_id,
                            concise_failure_reason(error)
                        ));
                    }
                }
                let response_tokenized = tool_response.with_multi_turn_chat_template::<M>(false);
                Ok(SegmentContent::ToolResponse(response_tokenized))
            } else {
                log_warning(format!(
                    "The model ended a sequence without a boxed answer or python tool call. Continuing generation. Flat id: {}",
                    question_flat_id
                ));
                generate_reasoning_or_tool_call_content::<M>(
                    question_flat_id,
                    trajectory,
                    llm_callable,
                    sglang_waiting_workers,
                    stop_signal,
                )
                .await
            }
        }
    }
}
