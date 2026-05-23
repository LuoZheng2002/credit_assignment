use std::collections::BTreeMap;

use reqwest::Client;
use research_utility::worker_message_tx::log_key_value_pair;

use crate::direct_tool::direct_trajectory::{DirectTrajectory, TrajectoryContent};
use crate::llm_model::MyTokenizer;
use crate::{
    agent::{
        state_to_actions::judge_answer_task, tool_call_execution::execute_planner_tool_call,
        trajectory_action_types::FinalAnswer, tree::CorrectnessJudgment,
    },
    direct_tool::{
        direct_tree::{DirectTree, SegmentContent, SegmentId},
        direct_tree_action::{DirectTreeAction, TokenPositionInTree},
        direct_tree_posterior::Posterior,
        direct_tree_status::DirectTreeStatus,
    },
    llm_model::{LlmCallable, LlmModelMarker, TokenLogprobCandidate, Top8Candidates},
};

#[derive(Debug)]
pub enum TokenViewMeta {
    NodeBranchingCandidate {
        parent_segment: SegmentId,
        child_segments: Vec<SegmentId>, // the children of parent segments, including self
        child_tokens: Vec<i32>, // the first token of each child segment, need to be excluded from choosing as the first token of the new branch
    },
    SegmentBranchingCandidate {
        segment_id: SegmentId,
        reasoning_only_segment_length: usize,
        first_half_length_after_split: usize,
        second_half_length_after_split: usize,
    },
}
// only provide necessary information for branching
#[derive(Debug)]
pub struct TokenViewForBranching<'a, M: LlmModelMarker> {
    pub token_position: TokenPositionInTree,
    pub token_id: i32,
    pub token_logprobs: Top8Candidates,
    pub token_meta: TokenViewMeta,
    pub corresponding_tree: &'a DirectTree<M>,
}

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn token_views_for_branching(&self) -> Vec<TokenViewForBranching<'_, M>> {
        // we need to find all the token views for branching, including both segment branching and node branching
        // for segment branching, we only consider the reasoning tokens, and we have the logprobs for those tokens
        // for node branching, we consider all the tokens in the segment, but we only have the logprobs for the first token (the branching token), and the rest of the tokens share the same logprobs as the branching token
        // for node branching, we also need to include the information of sibling segment id and parent segment id for calculating the uncertainty score
        let mut token_views: Vec<TokenViewForBranching<'_, M>> = Vec::new();
        for (segment_id, segment) in self.segments.iter() {
            let reasoning_only_tokens = segment.reasoning_only_tokens();
            for (token_index_in_reasoning, token_view) in reasoning_only_tokens.iter().enumerate() {
                assert!(token_view.flat_index == token_index_in_reasoning);
                let token_position = TokenPositionInTree {
                    segment_id: *segment_id,
                    content_index: token_view.content_index_in_segment,
                    offset: token_view.token_offset_in_content,
                };
                let token_meta = if token_index_in_reasoning == 0 {
                    // the first token in reasoning_only_tokens is considered to be a node branching candidate only if the first content in the segment is a reasoning content
                    // this is true if the first trunk has separate prompt segment and reasoning segment, and by induction the invariant should always hold
                    // so, we assert that the first token view corresponds to a reasoning content, and therefore it is a node branching candidate
                    assert!(token_view.content_index_in_segment == 0); // the first token view must correspond to the first content in the segment, which should be a reasoning content
                    // node branching candidate
                    let parent_segment = self.segments.get(segment_id).unwrap().parent_id.expect(
                        &format!("A reasoning only segment must have parents since prompt segment is root: {:?}", segment_id),
                    ); // considering that we want the invariant that every reasoning segment has a parent, we need to do special treatment for the first trunk
                    let child_segments = self.segments.get(segment_id).unwrap().child_ids.clone();
                    // sibling tokens are the first token view token of the sibling segments
                    let child_tokens = child_segments
                        .iter()
                        .map(|sibling_segment_id| {
                            self.segments
                                .get(sibling_segment_id)
                                .expect("Sibling segment id must exist in tree")
                                .first_reasoning_token()
                                .expect("Sibling segment must have at least one reasoning token")
                        })
                        .collect();
                    TokenViewMeta::NodeBranchingCandidate {
                        parent_segment,
                        child_segments,
                        child_tokens,
                    }
                } else {
                    // segment branching candidate
                    let reasoning_only_segment_length = reasoning_only_tokens.len();
                    let first_half_length_after_split = token_index_in_reasoning;
                    let second_half_length_after_split =
                        reasoning_only_segment_length - token_index_in_reasoning;
                    TokenViewMeta::SegmentBranchingCandidate {
                        segment_id: *segment_id,
                        reasoning_only_segment_length,
                        first_half_length_after_split,
                        second_half_length_after_split,
                    }
                };
                let token_view_for_branching = TokenViewForBranching {
                    token_position,
                    token_id: token_view.token,
                    token_logprobs: token_view.logprobs.clone(),
                    token_meta,
                    corresponding_tree: self,
                };
                token_views.push(token_view_for_branching);
            }
        }
        token_views
    }
    pub async fn produce_actions_from_direct_tree(
        &self,
        llm_callable: &M::Callable,
        client: Client,
        // rng: &mut StdRng,
    ) -> Vec<DirectTreeAction> {
        match self.status {
            DirectTreeStatus::CreatingTrunkTrajectory => {
                assert!(self.current_num_trunks < self.rollout_config.max_num_trunks);
                let root_id = self
                    .root_segment_id
                    .expect("Root segment id must exist when creating trunk trajectory");
                let (content_array, final_answer) = self
                    .generate_continuing_segment_contents(root_id, llm_callable)
                    .await;
                let correctness_judgment = judge_final_answer(
                    &final_answer,
                    &self.question.correct_answer,
                    &self.question.question,
                    client,
                )
                .await;
                log_key_value_pair(
                    "progress".into(),
                    format!(
                        "Question {}: Created and judged trunk trajectory, correctness: {}",
                        self.question.flat_id, correctness_judgment.is_correct
                    ),
                );
                vec![DirectTreeAction::CreateAndJudgeTrunkTrajectory {
                    content_array,
                    correctness_judgment,
                }]
            }
            DirectTreeStatus::CreatingOrChoosingBranchPoint => {
                assert!(
                    self.current_num_trunks == self.rollout_config.max_num_trunks,
                    "Current number of trunks must be equal to the max number of trunks before creating branch point"
                );
                assert!(!self.leaf_segment_judgments.is_empty());
                assert!(self.leaf_segment_judgments.len() < self.rollout_config.max_num_total_trajectories);

                let posteriors = self.calculate_segment_posteriors();
                let segment_uncertainty_scores =
                    self.posteriors_to_segment_uncertainty_scores(&posteriors);
                let token_views_for_branching = self.token_views_for_branching();

                let mut best_token_position: Option<TokenPositionInTree> = None;
                let mut best_token_id: Option<i32> = None;
                let mut best_branching_score = f32::NEG_INFINITY;
                let mut best_token_is_node: Option<bool> = None;

                assert!(!self.trunk_token_lengths.is_empty(), "Trunk token lengths must not be empty when creating or choosing branch point");
                let average_trunk_token_length = self
                    .trunk_token_lengths
                    .values()
                    .sum::<usize>() as f32
                    / self.trunk_token_lengths.len() as f32;
                log_key_value_pair("average_trunk_token_length".into(), average_trunk_token_length.to_string());

                for token_view in token_views_for_branching.iter() {
                    let segment_uncertainty_score = segment_uncertainty_scores
                        .get(&token_view.token_position.segment_id)
                        .expect("Each token view must correspond to a segment with an uncertainty score");
                    let (new_token_id, branching_score, token_is_node) =
                        match &token_view.token_meta {
                            TokenViewMeta::NodeBranchingCandidate {
                                child_segments,
                                child_tokens,
                                parent_segment,
                            } => {
                                let node_uncertainty_score =
                                    Self::node_uncertainty_score_from_parent_and_child_ids(
                                        *parent_segment,
                                        child_segments,
                                        &segment_uncertainty_scores,
                                    );
                                let Some((best_token_id, best_token_relative_probability)) =
                                    Self::best_token_and_relative_probability(
                                        &token_view.token_logprobs,
                                        child_tokens,
                                    )
                                else {
                                    continue; // if there is no valid branching token candidate, we skip this token view
                                };
                                let branching_factor = child_segments.len() as f32;
                                let branching_factor_penalty_multiplier =
                                    Self::branching_factor_penalty_multiplier(
                                        branching_factor as usize,
                                    );
                                let branching_score = node_uncertainty_score
                                    * best_token_relative_probability
                                    * branching_factor_penalty_multiplier;
                                (best_token_id, branching_score, true)
                            }
                            TokenViewMeta::SegmentBranchingCandidate {
                                segment_id,
                                reasoning_only_segment_length,
                                first_half_length_after_split,
                                second_half_length_after_split,
                            } => {
                                assert!(*segment_id == token_view.token_position.segment_id);
                                assert!(
                                    *reasoning_only_segment_length
                                        == *first_half_length_after_split
                                            + *second_half_length_after_split
                                );
                                // we already got segment_uncertainty_score
                                // then we need the relative probability
                                let Some((best_token_id, best_token_relative_probability)) =
                                    Self::best_token_and_relative_probability(
                                        &token_view.token_logprobs,
                                        &[token_view.token_id], // we exclude the existing token
                                    )
                                else {
                                    continue; // if there is no valid branching token candidate, we skip this token view
                                };
                                let segment_length_penalty_multiplier =
                                    Self::segment_length_penalty_multiplier(
                                        *first_half_length_after_split,
                                        *second_half_length_after_split,
                                        average_trunk_token_length,
                                    );
                                let branching_score = segment_uncertainty_score
                                    * best_token_relative_probability
                                    * segment_length_penalty_multiplier;
                                (best_token_id, branching_score, false)
                            }
                        };
                    if branching_score > best_branching_score {
                        best_branching_score = branching_score;
                        best_token_position = Some(token_view.token_position.clone());
                        best_token_id = Some(new_token_id);
                        best_token_is_node = Some(token_is_node);
                    }
                }
                if let Some(best_token_position) = best_token_position {
                    let new_branch_start_token = best_token_id
                        .expect("Best token id must exist if best token position exists");
                    let action = if best_token_is_node.unwrap() {
                        assert!(best_token_position.content_index == 0);
                        assert!(best_token_position.offset == 0);
                        DirectTreeAction::BranchFromNode {
                            position: best_token_position,
                            new_branch_start_token,
                        }
                    } else {
                        assert!(
                            best_token_position.content_index > 0 || best_token_position.offset > 0
                        );
                        DirectTreeAction::BranchFromSegment {
                            position: best_token_position,
                            new_branch_start_token,
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
                    .generate_continuing_segment_contents(focused_parent_segment_id, llm_callable)
                    .await;
                let correctness_judgment = judge_final_answer(
                    &final_answer,
                    &self.question.correct_answer,
                    &self.question.question,
                    client,
                )
                .await;
                let is_correct = correctness_judgment.is_correct;
                let action = DirectTreeAction::CreateAndJudgeBranchSegment {
                    contents: new_contents,
                    correctness_judgment,
                };
                log_key_value_pair(
                    "progress".into(),
                    format!(
                        "Question {}: Created and judged branch segment, correctness: {}",
                        self.question.flat_id, is_correct
                    ),
                );
                vec![action]
            }
            // DirectTreeStatus::JudgingBranchSegment => {
            //     //
            // }
            DirectTreeStatus::Complete => {
                // the tree is complete, no more actions can be taken
                unreachable!()
            }
        }
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
        let parent_uncertainty_score = segment_uncertainty_scores
            .get(&parent_segment_id)
            .expect("Parent segment must have an uncertainty score");
        let children_uncertainty_score: Vec<f32> = child_segment_ids
            .iter()
            .map(|child_id| {
                segment_uncertainty_scores
                    .get(child_id)
                    .expect("Child segment must have an uncertainty score")
            })
            .cloned()
            .collect();
        Self::node_uncertainty_score_from_parent_and_children(
            *parent_uncertainty_score,
            &children_uncertainty_score,
        )
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
    pub fn branching_factor_penalty_multiplier(branching_factor: usize) -> f32 {
        assert!(branching_factor >= 1);
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
        let shorter_half_length = std::cmp::min(first_half_length_after_split, second_half_length_after_split);
        let shorter_half_to_average_ratio = (shorter_half_length as f32) / average_trunk_token_length;
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
        // rng: &mut StdRng,
    ) -> (Vec<SegmentContent>, FinalAnswer) {
        let mut continuing_contents = Vec::new();
        loop {
            let trajectory = self.get_trajectory(target_segment_id, &continuing_contents);
            if let Some(answer) = trajectory.try_get_answer() {
                if continuing_contents.is_empty() {
                    println!("trajectory contents: {}", trajectory.to_decoded_string());
                    panic!(
                        "The trajectory should produce some continuing content before producing the answer"
                    );
                }
                return (continuing_contents, answer);
            }
            let next_content = generate_next_segment_content::<M>(&trajectory, llm_callable).await;
            continuing_contents.push(next_content.clone());
        }
    }
    pub fn trajectory_reasoning_token_length(&self, segment_id: SegmentId) -> usize {
        let mut trajectory_segments = Vec::new();
        let mut current_segment = Some(self.segments.get(&segment_id).expect("Segment id must exist in tree"));
        while let Some(segment) = current_segment {
            trajectory_segments.push(segment);
            if let Some(parent_id) = segment.parent_id {
                current_segment = self.segments.get(&parent_id);
            } else {
                break;
            }
        }
        trajectory_segments.iter().map(|segment| segment.reasoning_only_token_length()).sum()
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

fn direct_trajectory_to_prompt_tokens(trajectory: &DirectTrajectory) -> Vec<i32> {
    let mut prompt_tokens: Vec<i32> = Vec::new();
    for content in trajectory.trajectory_contents.iter() {
        let tokens = match content {
            TrajectoryContent::Prompt(tokens) => &tokens.tokens,
            TrajectoryContent::ReasoningOrToolCallIncomplete(tokens) => &tokens.tokens,
            TrajectoryContent::ReasoningOrToolCallComplete(tokens) => &tokens.tokens,
            TrajectoryContent::ToolResponse(tokens) => &tokens.tokens,
        };
        prompt_tokens.extend_from_slice(tokens);
    }
    prompt_tokens
}

async fn generate_reasoning_or_tool_call_content<M: LlmModelMarker>(
    // current_content: &[SegmentContent],
    trajectory: &DirectTrajectory,
    llm_callable: &M::Callable,
) -> SegmentContent {
    let prompt_tokens = direct_trajectory_to_prompt_tokens(trajectory);
    let response = llm_callable
        .generate_tokens_with_logprobs(prompt_tokens.clone(), true)
        .await;
    if response.tokens.is_empty() || response.decoded_string.trim().is_empty() {
        panic!(
            "LLM returned empty response. Decoded string: '{}', tokens: {:?}",
            response.decoded_string, response.tokens
        );
    }
    SegmentContent::ReasoningOrToolCall {
        tokens: response,
        complete: true,
    }
}

async fn generate_next_segment_content<M: LlmModelMarker>(
    trajectory: &DirectTrajectory,
    // current_content: &[SegmentContent],
    // client: Client,
    llm_callable: &M::Callable,
    // rng: &mut StdRng,
) -> SegmentContent {
    let last_trajectory_content = trajectory
        .trajectory_contents
        .last()
        .expect("Current content must not be empty");
    match last_trajectory_content {
        TrajectoryContent::Prompt(_)
        | TrajectoryContent::ToolResponse(_)
        | TrajectoryContent::ReasoningOrToolCallIncomplete(_) => {
            let new_content =
                generate_reasoning_or_tool_call_content::<M>(trajectory, llm_callable).await;
            new_content
        }
        TrajectoryContent::ReasoningOrToolCallComplete(_) => {
            let Some(tool_call) = trajectory.try_get_last_content_tool_call() else {
                println!("Trajectory contents: {}", trajectory.to_decoded_string());
                panic!(
                    "tool call should not be none when the last content is a reasoning or tool call content when generating next segment content"
                );
            };
            let tool_response = execute_planner_tool_call(&tool_call).await;
            let tool_response_raw = tool_response.to_raw_content();
            let response_tokenized = M::Tokenizer::tokenize(tool_response_raw);
            SegmentContent::ToolResponse(response_tokenized)
        }
    }
}

async fn judge_final_answer(
    final_answer: &FinalAnswer,
    correct_answer: &str,
    question: &str,
    client: Client,
) -> CorrectnessJudgment {
    let is_correct = match final_answer {
        FinalAnswer::ModelProvided(model_answer) => {
            judge_answer_task(
                model_answer.clone(),
                correct_answer.to_string(),
                question.to_string(),
                client,
            )
            .await
        }
        FinalAnswer::Failure(_error_message) => false,
    };
    CorrectnessJudgment {
        model_answer: final_answer.clone(),
        correct_answer: correct_answer.to_string(),
        is_correct,
    }
}
