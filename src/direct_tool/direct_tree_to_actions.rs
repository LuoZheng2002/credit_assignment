use std::collections::BTreeMap;

use rand::rngs::StdRng;
use reqwest::Client;

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentContent, SegmentId}, direct_tree_action::{BranchPosition, DirectTreeAction}, direct_tree_advantage::Posterior, direct_tree_status::DirectTreeStatus
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
                // advantage (mean / variance)
                // segment length
                // the best breaking point and its probability (with penalty of deviating from the middle point)
                
                // node advantage score is dependent on segment advantage score


                assert!(!self.root_segment_ids.is_empty());
                assert!(!self.leaf_segment_judgments.is_empty());

                let posteriors = self.calculate_segment_posteriors();
                let segment_branching_scores =
                    self.posteriors_to_segment_branching_scores(&posteriors);
                let node_branching_scores = self.segment_scores_to_node_scores(&segment_branching_scores);
                // let segment_branching_scores_with_penalty = self.add_penalty_to_segment_score(segment_branching_scores);
                // let node_branching_scores_with_penalty = self.add_penalty_to_node_score(node_branching_scores);
                
                let branching_candidates = sort_segment_or_node_candidates(
                    segment_branching_scores_with_penalty,
                    node_branching_scores_with_penalty,
                );
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
    pub fn segment_scores_to_node_scores(&self, segment_scores: &BTreeMap<SegmentId, f32>) -> BTreeMap<SegmentId, f32> {
        // each node is represented as the end of a segment
        self.segments
            .iter()
            .filter_map(|(segment_id, segment)| {
                let parent_score = *segment_scores
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
                        *segment_scores
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
    pub fn posteriors_to_segment_branching_scores(
        &self,
        posteriors: &BTreeMap<SegmentId, Posterior>,
    ) -> BTreeMap<SegmentId, SegmentBranchingScoreComposition> {
        let eps = 1e-8_f32;
        let alpha = 1.0_f32;
        // avoid division by zero
        if posteriors.is_empty() {
            return BTreeMap::new();
        }

        let means: Vec<f32> = posteriors.values().map(|posterior| posterior.mean).collect();
        let log_vars: Vec<f32> = posteriors
            .values()
            .map(|posterior| {
                let var = (2.0 * posterior.log_std).exp();
                (var + eps).ln()
            })
            .collect();

        let mean_of_means = means.iter().sum::<f32>() / means.len() as f32;
        let std_of_means = (means
            .iter()
            .map(|value| (value - mean_of_means).powi(2))
            .sum::<f32>()
            / means.len() as f32)
            .sqrt();

        let mean_of_log_vars = log_vars.iter().sum::<f32>() / log_vars.len() as f32;
        let std_of_log_vars = (log_vars
            .iter()
            .map(|value| (value - mean_of_log_vars).powi(2))
            .sum::<f32>()
            / log_vars.len() as f32)
            .sqrt();

        let uncertainty_scores = posteriors
            .iter()
            .map(|(segment_id, posterior)| {
                let mean_norm = (posterior.mean - mean_of_means) / (std_of_means + eps);

                let var = (2.0 * posterior.log_std).exp();
                let log_var = (var + eps).ln();
                let log_var_norm = (log_var - mean_of_log_vars) / (std_of_log_vars + eps);
                let var_norm = log_var_norm.exp();

                let ratio = mean_norm.abs() / (var_norm + eps).sqrt();
                let score = (-alpha * ratio.powi(2)).exp();
                (*segment_id, score)
            })
            .collect();
        todo!()
    }
    pub fn add_penalty_to_segment_score(&self, segment_scores: BTreeMap<SegmentId, f32>) -> BTreeMap<SegmentId, f32> {
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
    pub fn add_penalty_to_node_score(&self, node_scores: BTreeMap<SegmentId, f32>) -> BTreeMap<SegmentId, f32> {
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

#[derive(Debug, Clone)]
pub struct SegmentBranchingScoreComposition {
    pub uncertainty_score: f32, // ranges from 0 to 1, the higher the score, the more uncertain and the better for branching
    pub token_probability_multiplier: f32, // ranges from 0 to 1, equals to the probability of the new token at temperature 1.0
    pub first_half_length_penalty_multiplier: f32, // ranges from 0 to 1, the shorter the split first half is, the higher the penalty is, and the lower the multiplier is
    pub second_half_length_penalty_multiplier: f32, // ranges from 0 to 1, the shorter the split second half is, the higher the penalty is, and the lower the multiplier is
    pub chosen_branch_position: BranchPosition,
    pub chosen_token: i32,
}

#[derive(Debug, Clone)]
pub struct NodeBranchingScoreComposition {
    pub uncertainty_score: f32, // ranges from 0 to 1, the higher the score, the more uncertain and the better for branching
    pub token_probability_multiplier: f32, // ranges from 0 to 1, equals to the probability of the new token at temperature 1.0
    pub branching_factor_penalty_multiplier: f32, // ranges from 0 to 1, the larger the branching factor is, the higher the penalty is, and the lower the multiplier is
    pub chosen_token: i32,
}

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
)-> Vec<SegmentOrNodeCandidate> {
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
