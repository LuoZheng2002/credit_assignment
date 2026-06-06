use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use reqwest::Client;
use research_utility::progress_tui_server::log_warning;

use crate::atomic_count_guard::AtomicCountGuard;
use crate::direct_tool::direct_rollout::StopRequestedError;
use crate::direct_tool::direct_trajectory::{DirectTrajectory, TrajectoryContent};
use crate::direct_tool::direct_tree_action::DirectTreeAction::SubmitAnswer;
use crate::direct_tool::direct_tree_status::{
    GuidedBranchingSubStatus, SpontaneousBranchingSubStatus, TrunkSubStatus,
};
use crate::direct_tool::hybrid_dataset::{DatasetSplit, QuestionFlatId};
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

const ENABLE_FORCED_NEW_BRANCH_START_TOKEN: bool = false;

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

impl<'a, M: LlmModelMarker, S: DatasetSplit> DirectTree<'a, M, S> {
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

    pub async fn produce_action_from_direct_tree(
        &self,
        llm_callable: &M::Callable,
        client: Client,
        python_tool_pool: Arc<PythonToolServerPool>,
        sglang_waiting_workers: Arc<AtomicUsize>,
        judge_waiting_workers: Arc<AtomicUsize>,
        tool_waiting_workers: Arc<AtomicUsize>,
        stop_signal: Arc<AtomicBool>,
    ) -> Result<DirectTreeAction<M>, StopRequestedError> {
        let result = match &self.status {
            DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::CollectingSegmentContents {
                cumulative_content_array,
            }) => {
                self.produce_collecting_segment_action(
                    cumulative_content_array,
                    None,
                    llm_callable,
                    python_tool_pool,
                    sglang_waiting_workers,
                    tool_waiting_workers,
                    stop_signal,
                )
                .await?
            }
            DirectTreeStatus::WorkingOnGuidedBranching(
                GuidedBranchingSubStatus::CollectingSegmentContents {
                    cumulative_content_array,
                    new_branch_start_token,
                    ..
                },
            ) => {
                // Inject the branch-start token only on the first generation step of the new branch.
                let generation_start_token = cumulative_content_array
                    .is_empty()
                    .then_some(*new_branch_start_token);
                self.produce_collecting_segment_action(
                    cumulative_content_array,
                    generation_start_token,
                    llm_callable,
                    python_tool_pool,
                    sglang_waiting_workers,
                    tool_waiting_workers,
                    stop_signal,
                )
                .await?
            }
            DirectTreeStatus::WorkingOnSpontaneousBranching(
                SpontaneousBranchingSubStatus::CollectingSegmentContents {
                    cumulative_content_array,
                },
            ) => {
                self.produce_collecting_segment_action(
                    cumulative_content_array,
                    None,
                    llm_callable,
                    python_tool_pool,
                    sglang_waiting_workers,
                    tool_waiting_workers,
                    stop_signal,
                )
                .await?
            }
            DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::JudgingSegment {
                final_answer,
                ..
            })
            | DirectTreeStatus::WorkingOnGuidedBranching(
                GuidedBranchingSubStatus::JudgingSegment { final_answer, .. },
            )
            | DirectTreeStatus::WorkingOnSpontaneousBranching(
                SpontaneousBranchingSubStatus::JudgingSegment { final_answer, .. },
            ) => {
                let correctness_judgment = judge_final_answer(
                    &final_answer,
                    &self.action_log.question.correct_answer,
                    &self.action_log.question.question,
                    client,
                    JudgeAnswerModel::Gemini25FlashLite,
                    judge_waiting_workers,
                )
                .await;
                DirectTreeAction::JudgeAnswer(correctness_judgment)
            }
            DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::AttachingToTree {
                correctness_judgment,
                parent_segment_id,
                finalized_content_array,
            })
            | DirectTreeStatus::WorkingOnGuidedBranching(
                GuidedBranchingSubStatus::AttachingToTree {
                    correctness_judgment,
                    parent_segment_id,
                    finalized_content_array,
                },
            )
            | DirectTreeStatus::WorkingOnSpontaneousBranching(
                SpontaneousBranchingSubStatus::AttachingToTree {
                    correctness_judgment,
                    parent_segment_id,
                    prefix_trimmed_content_array: finalized_content_array,
                    ..
                },
            ) => DirectTreeAction::AttachSegmentToTree {
                parent_segment_id: *parent_segment_id,
                finalized_content_array: finalized_content_array.clone(),
                correctness_judgment: correctness_judgment.clone(),
            },
            DirectTreeStatus::WorkingOnGuidedBranching(
                GuidedBranchingSubStatus::SplittingTargetSegment {
                    position,
                    branch_from_node,
                    ..
                },
            )
            | DirectTreeStatus::WorkingOnSpontaneousBranching(
                SpontaneousBranchingSubStatus::SplittingTargetSegment {
                    position,
                    branch_from_node,
                    ..
                },
            ) => DirectTreeAction::SplitTreeSegment {
                position: position.clone(),
                branch_from_node: *branch_from_node,
            },
            DirectTreeStatus::WorkingOnGuidedBranching(
                GuidedBranchingSubStatus::DeterminingBranchingPoint,
            ) => self.determine_guided_branch_action(),
            DirectTreeStatus::WorkingOnSpontaneousBranching(
                SpontaneousBranchingSubStatus::DeterminingBranchingPoint {
                    finalized_content_array,
                    ..
                },
            ) => self.determine_spontaneous_branch_action(finalized_content_array),
            DirectTreeStatus::WorkingOnSpontaneousBranching(
                SpontaneousBranchingSubStatus::PrefixTrimmingNewSegment {
                    position: _,
                    position_in_segment,
                    finalized_content_array: _,
                    ..
                },
            ) => {
                let trim_position = position_in_segment.clone();
                DirectTreeAction::PrefixTrimNewSegment { trim_position }
            }
            DirectTreeStatus::Complete => {
                // the tree is complete, no more actions can be taken
                unreachable!()
            }
        };
        Ok(result)
    }

    async fn produce_collecting_segment_action(
        &self,
        cumulative_content_array: &[SegmentContent<M>],
        new_branch_start_token: Option<i32>,
        llm_callable: &M::Callable,
        python_tool_pool: Arc<PythonToolServerPool>,
        sglang_waiting_workers: Arc<AtomicUsize>,
        tool_waiting_workers: Arc<AtomicUsize>,
        stop_signal: Arc<AtomicBool>,
    ) -> Result<DirectTreeAction<M>, StopRequestedError> {
        let target_segment_id = self.root_segment_id.expect(
            "Root segment id must exist when creating trunk trajectory or spontaneous branch",
        );
        let trajectory = self.get_trajectory(target_segment_id, cumulative_content_array);
        match trajectory.try_get_answer() {
            Some(final_answer) => Ok(SubmitAnswer(final_answer)),
            None => {
                let generation_start_token = if cumulative_content_array.is_empty() {
                    new_branch_start_token
                } else {
                    None
                };
                let next_content = generate_next_segment_content(
                    self.action_log.question.flat_id,
                    &trajectory,
                    generation_start_token,
                    self.action_log.rollout_config.use_tool,
                    llm_callable,
                    python_tool_pool,
                    sglang_waiting_workers,
                    tool_waiting_workers,
                    stop_signal,
                )
                .await?;
                Ok(DirectTreeAction::AppendSegmentContent(next_content))
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
    // async fn generate_continuing_segment_contents(
    //     &self,
    //     // mut current_contents: Vec<SegmentContent>,
    //     target_segment_id: SegmentId,
    //     // client: Client,
    //     llm_callable: &M::Callable,
    //     python_tool_pool: Arc<PythonToolServerPool>,
    //     sglang_waiting_workers: Arc<AtomicUsize>,
    //     stop_signal: Arc<AtomicBool>,
    //     // rng: &mut StdRng,
    // ) -> Result<(Vec<SegmentContent<M>>, FinalAnswer), StopRequestedError> {
    //     let mut continuing_contents = Vec::new();
    //     loop {
    //         let trajectory = self.get_trajectory(target_segment_id, &continuing_contents);
    //         if let Some(answer) = trajectory.try_get_answer() {
    //             if continuing_contents.is_empty() {
    //                 log_error(format!(
    //                     "trajectory contents: {}",
    //                     trajectory.to_decoded_string()
    //                 ));
    //                 panic!(
    //                     "The trajectory should produce some continuing content before producing the answer"
    //                 );
    //             }
    //             return Ok((continuing_contents, answer));
    //         }
    //         let next_content = generate_next_segment_content::<M>(
    //             self.action_log.question.flat_id,
    //             &trajectory,
    //             llm_callable,
    //             python_tool_pool.clone(),
    //             sglang_waiting_workers.clone(),
    //             stop_signal.clone(),
    //         )
    //         .await?;
    //         if matches!(next_content, SegmentContent::ReasoningOrToolCall { .. }) {
    //             if let Some(SegmentContent::ReasoningOrToolCall { complete, .. }) =
    //                 continuing_contents.last_mut()
    //             {
    //                 if *complete {
    //                     *complete = false;
    //                 }
    //             }
    //         }
    //         continuing_contents.push(next_content);
    //     }
    // }
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

    fn determine_guided_branch_action(&self) -> DirectTreeAction<M> {
        assert!(!self.leaf_segment_judgments.is_empty());

        let posteriors = self.calculate_segment_posteriors(None);
        assert!(
            !posteriors.is_empty(),
            "Posteriors must not be empty when determining guided branching action"
        );
        let mut segment_uncertainty_scores =
            self.posteriors_to_segment_uncertainty_scores(&posteriors);
        for segment_id in self.segments.keys().copied() {
            segment_uncertainty_scores.entry(segment_id).or_insert(0.0);
        }
        let per_token_branching_scores =
            self.calculate_per_token_branching_scores(&segment_uncertainty_scores);

        let mut best_candidate: Option<(SegmentId, ContentIndex, usize, TokenBranchingScore)> =
            None;
        for (segment_id, content_map) in per_token_branching_scores {
            for (content_index, offset_map) in content_map {
                for (offset, token_score) in offset_map {
                    if !token_score.branching_score.is_finite() {
                        continue;
                    }
                    let replace = match best_candidate {
                        None => true,
                        Some((_, _, _, best)) => token_score.branching_score > best.branching_score,
                    };
                    if replace {
                        best_candidate = Some((segment_id, content_index, offset, token_score));
                    }
                }
            }
        }

        let Some((segment_id, content_index, offset, token_score)) = best_candidate else {
            return DirectTreeAction::NoAvailableBranchPoint;
        };
        if token_score.branching_score <= 0.0 {
            return DirectTreeAction::NoAvailableBranchPoint;
        }

        let (position, branch_from_node) = match token_score.branching_type {
            BranchingType::Node => (
                TokenPositionInTree {
                    segment_id,
                    content_index: 0,
                    offset: 0,
                },
                true,
            ),
            BranchingType::Segment => (
                TokenPositionInTree {
                    segment_id,
                    content_index,
                    offset,
                },
                false,
            ),
        };

        DirectTreeAction::BranchFromSegmentOrNodeGuided {
            position,
            new_branch_start_token: token_score.token_id,
            branch_from_node,
        }
    }

    fn determine_spontaneous_branch_action(
        &self,
        finalized_content_array: &[SegmentContent<M>],
    ) -> DirectTreeAction<M> {
        if finalized_content_array.is_empty() {
            return DirectTreeAction::NoAvailableBranchPoint;
        }
        let prefix_result = self.find_longest_common_prefix(finalized_content_array);

        let position = prefix_result.diverge_position_in_tree;
        let branch_from_node = position.content_index == 0 && position.offset == 0;

        DirectTreeAction::BranchFromSegmentOrNodeSpontaneous {
            position,
            branch_from_node,
            position_in_segment: prefix_result.diverge_position_in_query,
        }
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

fn is_context_length_exceeded_error(error: &str) -> bool {
    error
        .to_ascii_lowercase()
        .contains("context length exceeded")
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

fn concise_model_response(response: &str) -> String {
    const MAX_RESPONSE_CHARS: usize = 40;
    const EDGE_RESPONSE_CHARS: usize = 20;

    let response_len = response.chars().count();
    if response_len <= MAX_RESPONSE_CHARS {
        return response.to_string();
    }

    let prefix: String = response.chars().take(EDGE_RESPONSE_CHARS).collect();
    let suffix: String = response
        .chars()
        .skip(response_len - EDGE_RESPONSE_CHARS)
        .collect();
    format!("{}...{}", prefix, suffix)
}

async fn generate_reasoning_or_tool_call_content<M: LlmModelMarker, S: DatasetSplit>(
    _question_flat_id: QuestionFlatId<S>,
    trajectory: &DirectTrajectory<M>,
    new_branch_start_token: Option<i32>,
    use_tool: bool,
    llm_callable: &M::Callable,
    sglang_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
) -> Result<SegmentContent<M>, StopRequestedError> {
    let mut prompt_tokens = trajectory.to_prompt_tokens();
    if let Some(start_token) = new_branch_start_token {
        prompt_tokens.push(start_token);
    }
    let mut response = None;
    let mut last_error: Option<String> = None;
    for trial in 1..=3 {
        let _num_sglang_waiting_workers_guard = AtomicCountGuard::new(
            sglang_waiting_workers.clone(),
            "sglang_waiting_workers".to_string(),
        );
        if stop_signal.load(Ordering::Relaxed) {
            return Err(StopRequestedError);
        }
        let generation_result = llm_callable
            .generate_tokens_with_logprobs(prompt_tokens.clone(), use_tool, 1.0, true)
            .await;
        match generation_result {
            Ok(result) => {
                response = Some(result);
                break;
            }
            Err(error) => {
                if is_context_length_exceeded_error(&error) {
                    log_warning(format!(
                        "generate_tokens_with_logprobs hit context limit on trial {}/3; using synthetic EOS response without retry: {}",
                        trial, error
                    ));
                    response = Some(single_eos_response::<M>());
                    break;
                }
                last_error = Some(error.clone());
                log_warning(format!(
                    "generate_tokens_with_logprobs failed on trial {}/3: {}",
                    trial, error
                ));
            }
        }
    }

    let mut response = response.unwrap_or_else(|| {
        panic!(
            "generate_tokens_with_logprobs failed after 3 trials. Last error: {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        )
    });
    let decoded_response = response.decode();
    if response.tokens.is_empty() {
        panic!(
            "LLM returned empty response. Decoded string: '{}', tokens: {:?}",
            decoded_response, response.tokens
        );
    }
    if let Some(start_token) = new_branch_start_token {
        response.tokens.insert(0, start_token);
        response
            .logprobs
            .insert(0, fallback_top8_for_token(start_token));
    }
    let result = SegmentContent::ReasoningOrToolCall {
        tokens: response,
        complete: true,
    };
    Ok(result)
}

async fn generate_next_segment_content<M: LlmModelMarker, S: DatasetSplit>(
    question_flat_id: QuestionFlatId<S>,
    trajectory: &DirectTrajectory<M>,
    new_branch_start_token: Option<i32>,
    use_tool: bool,
    // current_content: &[SegmentContent],
    // client: Client,
    llm_callable: &M::Callable,
    python_tool_pool: Arc<PythonToolServerPool>,
    sglang_waiting_workers: Arc<AtomicUsize>,
    tool_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
    // rng: &mut StdRng,
) -> Result<SegmentContent<M>, StopRequestedError> {
    let new_branch_start_token = if ENABLE_FORCED_NEW_BRANCH_START_TOKEN {
        new_branch_start_token
    } else {
        None
    };
    let last_trajectory_content = trajectory
        .trajectory_contents
        .last()
        .expect("Current content must not be empty");
    match last_trajectory_content {
        TrajectoryContent::Prompt(_)
        | TrajectoryContent::ToolResponse(_)
        | TrajectoryContent::ReasoningOrToolCallIncomplete(_) => {
            let new_content = generate_reasoning_or_tool_call_content::<M, S>(
                question_flat_id,
                trajectory,
                new_branch_start_token,
                use_tool,
                llm_callable,
                sglang_waiting_workers,
                stop_signal,
            )
            .await?;
            Ok(new_content)
        }
        TrajectoryContent::ReasoningOrToolCallComplete(tokens) => {
            if let Some(tool_call) = trajectory.try_get_last_content_tool_call() {
                // log_key_value_pair("info".to_string(), "Executing a tool call".to_string());
                let _num_tool_waiting_workers_guard =
                    AtomicCountGuard::new(tool_waiting_workers, "tool_waiting_workers".to_string());
                let tool_response = execute_python_tool_call(&python_tool_pool, &tool_call).await;
                match &tool_response {
                    PythonToolResponse::PythonSuccess(_) => {}
                    PythonToolResponse::PythonError(error) => {
                        log_warning(format!(
                            "Tool call failed. flat_id={:?} reason={}",
                            question_flat_id,
                            concise_failure_reason(error)
                        ));
                    }
                }
                let response_tokenized = tool_response.with_multi_turn_chat_template::<M>(false);
                Ok(SegmentContent::ToolResponse(response_tokenized))
            } else {
                let response = tokens.decode();
                let concise_response = concise_model_response(&response);
                log_warning(format!(
                    "The model ended a sequence without a boxed answer or python tool call. Flat id: {:?}. Response: {:?}",
                    question_flat_id, concise_response
                ));
                generate_reasoning_or_tool_call_content::<M, S>(
                    question_flat_id,
                    trajectory,
                    new_branch_start_token,
                    use_tool,
                    llm_callable,
                    sglang_waiting_workers,
                    stop_signal,
                )
                .await
            }
        }
    }
}
