use std::collections::BTreeMap;

use rand::rngs::StdRng;
use reqwest::Client;

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentContent, SegmentId},
        direct_tree_action::{DirectTreeAction, TokenPositionInTree},
        direct_tree_advantage::Posterior,
        direct_tree_status::DirectTreeStatus,
    },
    llm_model::{LlmModelMarker, TokenLogprobCandidate, Top8Candidates},
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
                        "A reasoning only token must have parents since prompt segment is root",
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
        rng: &mut StdRng,
    ) -> Vec<DirectTreeAction> {
        match self.status {
            DirectTreeStatus::CreatingTrunkTrajectory => {
                assert!(self.root_segment_ids.len() < self.num_trunks);
                let content_array = generate_continuing_segment_contents::<M>(
                    Vec::new(),
                    client,
                    llm_callable,
                    rng,
                )
                .await;
                vec![DirectTreeAction::CreateAndFocusTrunkTrajectory { content_array }]
            }
            DirectTreeStatus::CreatingOrChoosingBranchPoint => {
                assert!(!self.root_segment_ids.is_empty());
                assert!(!self.leaf_segment_judgments.is_empty());

                let posteriors = self.calculate_segment_posteriors();
                let segment_uncertainty_scores =
                    self.posteriors_to_segment_uncertainty_scores(&posteriors);
                let token_views_for_branching = self.token_views_for_branching();
                // now we consider the branching position at the token level granularity
                // each segment can have one node branching candidate and multiple segment branching candidates
                // actually the asymptotic complexity is the same

                let mut best_token_position: Option<TokenPositionInTree> = None;
                let mut best_token_id: Option<i32> = None;
                let mut best_branching_score = f32::NEG_INFINITY;
                let mut best_token_is_node: Option<bool> = None;
                // we also need to pass in the token id for branching

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
                // we are currently creating a branch segment, so the action should be to create and focus on a new branch segment under the current branch point
                todo!()
            }
            DirectTreeStatus::JudgingBranchSegment => {
                // we are currently judging a branch segment, so the action should be to judge the correctness of the focused segment
                todo!()
            }
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
        let k_branching_factor = 0.35_f32;
        (-k_branching_factor * (branching_factor as f32 - 1.0)).exp()
    }

    pub fn segment_length_penalty_multiplier(
        first_half_length_after_split: usize,
        second_half_length_after_split: usize,
    ) -> f32 {
        assert!(first_half_length_after_split > 0);
        assert!(second_half_length_after_split > 0);
        // the same exponential falloff formula as add_penalty_to_segment_score
        let k_length = 0.35_f32;
        let first_half_length_penalty_multiplier =
            1.0 - (-k_length * (first_half_length_after_split as f32 - 1.0)).exp();
        let second_half_length_penalty_multiplier =
            1.0 - (-k_length * (second_half_length_after_split as f32 - 1.0)).exp();
        first_half_length_penalty_multiplier * second_half_length_penalty_multiplier
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

pub enum SegmentContentResult {
    Continue(SegmentContent),
    Stop(SegmentContent),
    Error(String),
}

// when can a trajectory end?
// 1. found answer in \boxed{}
// 2. context length exceeded
// 3. other scenarios that require termination
async fn generate_next_segment_content<M: LlmModelMarker>(
    current_content: &[SegmentContent],
    client: Client,
    llm_callable: &M::Callable,
    rng: &mut StdRng,
) -> SegmentContentResult {
    // this function generates the content for the next segment to be added to the tree, based on the current tree structure and focused segment
    todo!()
}

async fn generate_continuing_segment_contents<M: LlmModelMarker>(
    mut current_content: Vec<SegmentContent>,
    client: Client,
    llm_callable: &M::Callable,
    rng: &mut StdRng,
) -> Vec<SegmentContent> {
    let mut continuing_contents = Vec::new();
    loop {
        let next_content_result =
            generate_next_segment_content::<M>(&current_content, client.clone(), llm_callable, rng)
                .await;
        match next_content_result {
            SegmentContentResult::Continue(next_content) => {
                current_content.push(next_content.clone());
                continuing_contents.push(next_content);
            }
            SegmentContentResult::Stop(next_content) => {
                current_content.push(next_content.clone());
                continuing_contents.push(next_content);
                break;
            }
            SegmentContentResult::Error(error_message) => {
                println!(
                    "Error generating segment content: {}, stopping generation for this trajectory",
                    error_message
                );
                break;
            }
        }
    }
    continuing_contents
}
