use rand::rngs::StdRng;
use reqwest::Client;

use crate::{
    direct_tool::{
        direct_tree::{DirectTree, SegmentContent},
        direct_tree_action::DirectTreeAction,
        direct_tree_status::DirectTreeStatus,
    },
    llm_model::LlmModelMarker,
};

pub async fn produce_actions_from_direct_tree<M: LlmModelMarker>(
    direct_tree: &DirectTree<M>,
    llm_callable: &M::Callable,
    client: Client,
    rng: &mut StdRng,
) -> Vec<DirectTreeAction> {
    match direct_tree.status {
        DirectTreeStatus::CreatingTrunkTrajectory => {
            assert!(direct_tree.root_segment_ids.len() < direct_tree.num_trunks);
            let mut content = Vec::new();
            vec![DirectTreeAction::CreateAndFocusTrunkTrajectory { content }]
        }
        DirectTreeStatus::CreatingOrChoosingBranchPoint => {
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

pub enum SegmentContentResult {
    Continue(SegmentContent),
    Stop(SegmentContent),
    Error(String),
}

// when can a trajectory end?
// 1. found answer in \boxed{}
// 2. context length exceeded
// 3. other scenarios that require termination
pub fn generate_next_segment_content<M: LlmModelMarker>(
    current_content: &[SegmentContent],
    client: Client,
    llm_callable: &M::Callable,
    rng: &mut StdRng,
) -> SegmentContentResult {
    // this function generates the content for the next segment to be added to the tree, based on the current tree structure and focused segment
    todo!()
}
