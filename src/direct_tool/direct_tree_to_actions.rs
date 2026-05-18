use std::collections::BTreeMap;

use rand::rngs::StdRng;
use reqwest::Client;

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentContent, SegmentId},
        direct_tree_action::DirectTreeAction,
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
                assert!(!self.root_segment_ids.is_empty());
                assert!(!self.leaf_segment_judgments.is_empty());

                let posteriors = self.calculate_segment_posteriors();
                let segment_branching_scores =
                    self.posteriors_to_segment_branching_scores(&posteriors);
                let node_branching_scores = self.segment_scores_to_node_scores(&segment_branching_scores);
                let segment_branching_scores_with_penalty = self.add_penalty_to_segment_score(segment_branching_scores);
                let node_branching_scores_with_penalty = self.add_penalty_to_node_score(node_branching_scores);
                // 
                let branching_candidates = sort_segment_or_node_candidates(
                    segment_branching_scores_with_penalty,
                    node_branching_scores_with_penalty,
                );
                
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
        posteriors: &BTreeMap<SegmentId, crate::direct_tool::direct_tree_advantage::Posterior>,
    ) -> BTreeMap<SegmentId, f32> {
        let eps = 1e-8_f32;
        let alpha = 1.0_f32;

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

        posteriors
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
            .collect()
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
pub enum SegmentOrNodeCandidate {
    Segment(SegmentId),
    Node(SegmentId),
}

#[derive(Debug, Clone)]
pub struct SegmentOrNodeCandidateWithScore {
    pub candidate: SegmentOrNodeCandidate,
    pub score: f32,
}

pub fn sort_segment_or_node_candidates(
    segment_scores: BTreeMap<SegmentId, f32>,
    node_scores: BTreeMap<SegmentId, f32>,
) -> Vec<SegmentOrNodeCandidateWithScore> {
    let mut candidates_with_scores = Vec::new();
    for (segment_id, score) in segment_scores {
        candidates_with_scores.push(SegmentOrNodeCandidateWithScore {
            candidate: SegmentOrNodeCandidate::Segment(segment_id),
            score,
        });
    }
    for (node_id, score) in node_scores {
        candidates_with_scores.push(SegmentOrNodeCandidateWithScore {
            candidate: SegmentOrNodeCandidate::Node(node_id),
            score,
        });
    }
    // descending order: larger score means higher branch priority
    candidates_with_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    candidates_with_scores
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
