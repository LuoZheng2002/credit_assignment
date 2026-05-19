use std::collections::{BTreeMap, HashMap};

use rand::rngs::StdRng;
use reqwest::Client;

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentContent, SegmentId},
        direct_tree_action::{BranchPosition, DirectTreeAction},
        direct_tree_advantage::Posterior,
        direct_tree_status::DirectTreeStatus,
    },
    llm_model::LlmModelMarker,
};

impl<M: LlmModelMarker> DirectTree<M> {
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
                // segment score composition:
                // advantage (mean / std)
                // segment length
                // the best breaking point and its probability (with penalty of deviating from the middle point)

                // node advantage score is dependent on segment advantage score

                assert!(!self.root_segment_ids.is_empty());
                assert!(!self.leaf_segment_judgments.is_empty());

                let posteriors = self.calculate_segment_posteriors();
                // let segment_branching_scores =
                //     self.posteriors_to_segment_branching_scores(&posteriors);
                let segment_uncertainty_scores =
                    self.posteriors_to_segment_uncertainty_scores(&posteriors);
                let segment_probability_and_length_scores =
                    self.segment_probability_and_length_scores();
                let node_branching_scores = self
                    .segment_uncertainty_scores_to_node_uncertainty_scores(
                        &segment_uncertainty_scores,
                    );
                let node_probability_and_branching_factor_scores =
                    self.node_probability_and_branching_factor_scores();
                // let segment_branching_scores_with_penalty = self.add_penalty_to_segment_score(segment_branching_scores);
                // let node_branching_scores_with_penalty = self.add_penalty_to_node_score(node_branching_scores);

                // let branching_candidates = sort_segment_or_node_candidates(
                //     segment_branching_scores_with_penalty,
                //     node_branching_scores_with_penalty,
                // );
                // branching candidates are sorted in a descending order

                // we are currently creating or choosing a branch point, so the action could be either to create a new branch point or to move to an existing branch point
                todo!()
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
    pub fn segment_uncertainty_scores_to_node_uncertainty_scores(
        &self,
        segment_uncertainty_scores: &BTreeMap<SegmentId, f32>,
    ) -> BTreeMap<SegmentId, f32> {
        // each node is represented as the end of a segment
        self.segments
            .iter()
            .filter_map(|(segment_id, segment)| {
                let parent_score = *segment_uncertainty_scores
                    .get(segment_id)
                    .expect("Each segment must have a segment score");

                let branching_factor = segment.child_ids.len();
                if branching_factor == 0 {
                    return None;
                }

                let child_score_sum: f32 = segment
                    .child_ids
                    .iter()
                    .map(|child_id| {
                        *segment_uncertainty_scores
                            .get(child_id)
                            .expect("Each child segment must have a segment score")
                    })
                    .sum();

                let b = branching_factor as f32;
                let node_score = (b * parent_score + child_score_sum) / (2.0 * b);
                Some((*segment_id, node_score))
            })
            .collect()
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

    pub fn segment_probability_and_length_scores(
        &self,
    ) -> BTreeMap<SegmentId, SegmentBranchPositionScore> {
        // we need to convert a segment to a reasoning-only segment, find the best token, and has to map back
        let mut branch_positions: BTreeMap<SegmentId, SegmentBranchPositionScore> = BTreeMap::new();
        for (segment_id, segment) in &self.segments {
            let reasoning_only_tokens = segment.reasoning_only_tokens();
            let length = reasoning_only_tokens.len();
            let mut best_branch_position_candidate: Option<SegmentBranchPositionScore> = None;
            for token_view in reasoning_only_tokens.iter() {
                // actually we do not consider distance to middle, but the two split lengths
                let first_half_length = token_view.flat_index as f32;
                let second_half_length = (length - token_view.flat_index) as f32;
                // the same exponential falloff formula as add_penalty_to_segment_score
                let k_length = 0.35_f32;
                let first_half_length_penalty_multiplier =
                    1.0 - (-k_length * (first_half_length - 1.0)).exp();
                let second_half_length_penalty_multiplier =
                    1.0 - (-k_length * (second_half_length - 1.0)).exp();
                // if token probability is too low, we do not consider this branching position candidate
                // we use relative probability for penalty and pruning
                let max_logprob = token_view
                    .logprobs
                    .iter()
                    .map(|candidate| candidate.logprob)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut token_relative_probabilities: BTreeMap<i32, f32> = token_view
                    .logprobs
                    .iter()
                    .map(|candidate| {
                        let relative_probability = (candidate.logprob - max_logprob).exp(); // convert logprob to probability and normalize by max_logprob for numerical stability
                        (candidate.token_id, relative_probability)
                    })
                    .collect();
                let existing_token = token_view.token;
                token_relative_probabilities.remove(&existing_token);
                assert!(
                    token_relative_probabilities.len() >= 7,
                    "There should be at least 7 candidate tokens other than the existing token"
                );
                let (best_token_id, best_token_relative_probability) = token_relative_probabilities
                    .into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .expect("There should be at least one candidate token");
                if best_token_relative_probability < 0.1 {
                    // if the best token has very low probability, we do not consider this branching position candidate
                    continue;
                }
                let branch_position_candidate = SegmentBranchPositionScore {
                    token_id: best_token_id,
                    token_relative_probability: best_token_relative_probability,
                    first_half_length_penalty_multiplier,
                    second_half_length_penalty_multiplier,
                    position: BranchPosition {
                        content_index: token_view.original_content_index,
                        offset: token_view.original_token_offset,
                    },
                };
                if best_branch_position_candidate.is_none()
                    || branch_position_candidate.total_penalty_multiplier()
                        > best_branch_position_candidate
                            .as_ref()
                            .unwrap()
                            .total_penalty_multiplier()
                {
                    best_branch_position_candidate = Some(branch_position_candidate);
                }
            }
            if let Some(best_branch_position_candidate) = best_branch_position_candidate {
                branch_positions.insert(*segment_id, best_branch_position_candidate);
            } else {
                println!(
                    "Warning: No valid branching position candidate found for segment {:?}, this segment will not be branched",
                    segment_id
                );
            }
        }
        branch_positions
    }
    pub fn node_probability_and_branching_factor_scores(
        &self,
    ) -> BTreeMap<SegmentId, NodeProbabilityAndBranchingFactorScore> {
        let mut node_scores: BTreeMap<SegmentId, NodeProbabilityAndBranchingFactorScore> =
            BTreeMap::new();
        for (segment_id, segment) in &self.segments {
            let branching_factor = segment.child_ids.len();
            if branching_factor == 0 {
                continue;
            }
            let k_branching_factor = 0.35_f32;
            let branching_factor_penalty_multiplier =
                (-k_branching_factor * (branching_factor as f32 - 1.0)).exp();

            // let node_score = NodeProbabilityAndBranchingFactorScore {
            //     branching_factor_penalty_multiplier,
            //     chosen_token: best_branch_position_score.token_id,
            // };
            // node_scores.insert(*segment_id, node_score);
            todo!() // we need to find the token probability for the new token
        }
        node_scores
    }
    pub fn add_penalty_to_segment_score(
        &self,
        segment_scores: BTreeMap<SegmentId, f32>,
    ) -> BTreeMap<SegmentId, f32> {
        // the shorter the segment is (only consider the reasoning tokens), the higher the penalty is
        let k_segment = 0.35_f32;

        let reasoning_lengths: BTreeMap<SegmentId, usize> = self
            .segments
            .iter()
            .map(|(segment_id, segment)| {
                let length = segment
                    .content
                    .iter()
                    .map(|content| match content {
                        SegmentContent::ReasoningOrToolCall(tokens) => tokens.tokens.len(),
                        SegmentContent::Prompt(_) | SegmentContent::ToolResponse(_) => 0,
                    })
                    .sum::<usize>();
                (*segment_id, length)
            })
            .collect();

        segment_scores
            .into_iter()
            .map(|(segment_id, score)| {
                let len = *reasoning_lengths
                    .get(&segment_id)
                    .expect("Segment score id must exist in tree");
                assert!(len >= 1, "Segment reasoning length must be >= 1");
                let len = len as f32;
                let multiplier = 1.0 - (-k_segment * (len - 1.0)).exp();
                (segment_id, score * multiplier)
            })
            .collect()
    }
    pub fn add_penalty_to_node_score(
        &self,
        node_scores: BTreeMap<SegmentId, f32>,
    ) -> BTreeMap<SegmentId, f32> {
        // the larger the branching factor is, the higher the penalty is
        let k_node = 0.35_f32;

        node_scores
            .into_iter()
            .map(|(node_id, score)| {
                let branching_factor = self
                    .segments
                    .get(&node_id)
                    .expect("Node score id must exist in tree")
                    .child_ids
                    .len();
                assert!(branching_factor >= 1, "Node branching factor must be >= 1");
                let branch_factor = branching_factor as f32;
                let multiplier = (-k_node * (branch_factor - 1.0)).exp();
                (node_id, score * multiplier)
            })
            .collect()
    }
}

// #[derive(Debug, Clone)]
// pub struct SegmentBranchingScoreComposition {
//     pub uncertainty_score: f32, // ranges from 0 to 1, the higher the score, the more uncertain and the better for branching
//     pub branch_position: SegmentBranchPositionScore, // the best branching position in the segment and its corresponding token probability multiplier
// }
#[derive(Debug, Clone)]
pub struct SegmentBranchPositionScore {
    pub position: BranchPosition,
    pub token_id: i32,
    pub token_relative_probability: f32, // ranges from 0 to 1, equals to the probability of the new token at temperature 1.0
    pub first_half_length_penalty_multiplier: f32, // ranges from 0 to 1, the shorter the split first half is, the higher the penalty is, and the lower the multiplier is
    pub second_half_length_penalty_multiplier: f32, // ranges from 0 to 1, the shorter the split second half is, the higher the penalty is, and the lower the multiplier is
}

impl SegmentBranchPositionScore {
    pub fn total_penalty_multiplier(&self) -> f32 {
        self.token_relative_probability
            * self.first_half_length_penalty_multiplier
            * self.second_half_length_penalty_multiplier
    }
}

#[derive(Debug, Clone)]
pub struct NodeProbabilityAndBranchingFactorScore {
    pub token_relative_probability: f32, // ranges from 0 to 1, equals to the probability of the new token at temperature 1.0
    pub branching_factor_penalty_multiplier: f32, // ranges from 0 to 1, the larger the branching factor is, the higher the penalty is, and the lower the multiplier is
    pub chosen_token: i32,
}

// #[derive(Debug, Clone)]
// pub struct NodeBranchingScoreComposition {
//     pub uncertainty_score: f32, // ranges from 0 to 1, the higher the score, the more uncertain and the better for branching
//     pub token_relative_probability: f32, // ranges from 0 to 1, equals to the probability of the new token at temperature 1.0
//     pub branching_factor_penalty_multiplier: f32, // ranges from 0 to 1, the larger the branching factor is, the higher the penalty is, and the lower the multiplier is
//     pub chosen_token: i32,
// }

// #[derive(Debug, Clone)]
// pub enum BranchingPoint {
//     Segment {
//         segment_id: SegmentId,
//         position: BranchPosition,
//     },
//     Node(SegmentId),
// }
#[derive(Debug, Clone)]
pub struct SegmentCandidate {
    pub segment_id: SegmentId,
    pub position: BranchPosition,
    pub score: f32,
}
#[derive(Debug, Clone)]
pub struct NodeCandidate {
    pub node_id: SegmentId,
    pub score: f32,
}
pub enum SegmentOrNodeCandidate {
    Segment(SegmentCandidate),
    Node(NodeCandidate),
}
impl SegmentOrNodeCandidate {
    pub fn score(&self) -> f32 {
        match self {
            SegmentOrNodeCandidate::Segment(candidate) => candidate.score,
            SegmentOrNodeCandidate::Node(candidate) => candidate.score,
        }
    }
}

pub fn sort_segment_or_node_candidates(
    segment_scores: BTreeMap<SegmentId, SegmentCandidate>,
    node_scores: BTreeMap<SegmentId, NodeCandidate>,
) -> Vec<SegmentOrNodeCandidate> {
    let mut candidates = Vec::new();
    for (_segment_id, candidate) in segment_scores {
        candidates.push(SegmentOrNodeCandidate::Segment(candidate));
    }
    for (_node_id, candidate) in node_scores {
        candidates.push(SegmentOrNodeCandidate::Node(candidate));
    }
    // descending order: larger score means higher branch priority
    candidates.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());
    candidates
}

// pub fn pick_best_branching_point(candidates: Vec<SegmentOrNodeCandidate>) -> Option<BranchingPoint> {
//     candidates.into_iter().next().map(|candidate| match candidate {
//         SegmentOrNodeCandidate::Segment(segment_candidate) => BranchingPoint::Segment {
//             segment_id: segment_candidate.segment_id,
//             position: segment_candidate.position,
//         },
//         SegmentOrNodeCandidate::Node(node_candidate) => BranchingPoint::Node(node_candidate.node_id),
//     })
// }

fn try_get_best_branch_position_in_segment(segment: &SegmentContent) -> Option<BranchPosition> {
    // ideally we need to choose the middle point of the segment
    // but the middle point of the segment might have very concentrated probability distribution
    // in this case if we choose the second option of the starting token, this will confuse the model
    // instead, we need to give a comprehensive score for each token in the segment.
    // the primary factor is the probability of the new token to be chosen

    // in fact, for determining the branching point, we also need to consider the probability of the new token
    // then branching from existing branching point may be very unfair compared with branching from segments

    // if each segment is to be chosen, then it will have its best score and best branching position
    todo!()
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
