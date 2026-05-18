use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    agent::tree::CorrectnessJudgment,
    direct_tool::{direct_tree_action::DirectTreeAction, direct_tree_status::DirectTreeStatus},
    llm_model::{LlmModelMarker, MyTokenizer, TokenArrayWithLogprob},
    token_array::TokenArray,
};

// this tree is similar to the completed tree in src/agent folder, but now it runs on a lightweight tool-calling context instead of a heavy agent framework
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTree<M: LlmModelMarker> {
    pub flat_id: usize, // the same flat id as the one in the hybrid dataset
    pub dataset_name: String,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
    // states
    pub status: DirectTreeStatus,
    pub segments: BTreeMap<usize, Segment>, // segment_id -> segment. A segment branched from the middle is destroyed and its id is not reused to avoid hiding sneaky bugs
    pub root_segment_ids: Vec<usize>,
    pub leaf_segment_judgments: BTreeMap<usize, CorrectnessJudgment>,
    pub next_segment_id: usize,
    pub focused_segment_id: Option<usize>, // the segment after which we create a new branch and rollout until finding the answer
    pub completed: bool,
    // hyperparameters
    pub num_trunks: usize,
    pub max_num_total_trajectories: usize,
    pub use_tool: bool,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<M>, // for tokenizer utility
}

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn new(
        flat_id: usize,
        dataset_name: String,
        question_id: usize,
        question: String,
        correct_answer: String,
        num_trunks: usize,
        max_num_total_trajectories: usize,
        use_tool: bool,
    ) -> Self {
        Self {
            flat_id,
            dataset_name,
            question_id,
            question,
            correct_answer,
            status: DirectTreeStatus::CreatingTrunkTrajectory,
            segments: BTreeMap::new(),
            root_segment_ids: vec![],
            leaf_segment_judgments: BTreeMap::new(),
            next_segment_id: 0,
            focused_segment_id: None,
            completed: false,
            num_trunks,
            max_num_total_trajectories,
            use_tool,
            _phantom: std::marker::PhantomData::<M>,
        }
    }
    pub fn apply_action(&mut self, action: DirectTreeAction) {
        match action {
            DirectTreeAction::CreateAndFocusTrunkTrajectory { content } => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingTrunkTrajectory
                ));
                let segment_id = self.next_segment_id;
                self.next_segment_id += 1;
                self.segments.insert(
                    segment_id,
                    Segment {
                        segment_id,
                        content,
                        child_ids: vec![],
                        parent_id: None,
                    },
                );
                self.root_segment_ids.push(segment_id);
                self.focused_segment_id = Some(segment_id);
                // update status
                if self.root_segment_ids.len() >= self.num_trunks {
                    self.status = DirectTreeStatus::CreatingOrChoosingBranchPoint;
                } else {
                    self.status = DirectTreeStatus::CreatingTrunkTrajectory;
                }
            }
            DirectTreeAction::CreateAndMoveToBranchPoint {
                target_segment_id,
                position,
            } => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingOrChoosingBranchPoint
                ));
                let new_first_half_id = self.next_segment_id;
                self.next_segment_id += 1;
                let new_second_half_id = self.next_segment_id;
                self.next_segment_id += 1;
                let target_segment = self
                    .segments
                    .remove(&target_segment_id)
                    .expect("Target segment id must exist");
                let target_content = &target_segment.content[position.content_index];
                let mut first_half_content_array =
                    target_segment.content[..position.content_index].to_vec();
                let mut second_half_content_array =
                    target_segment.content[position.content_index + 1..].to_vec();
                let SegmentContent::ReasoningOrToolCall(original_tokens) = target_content else {
                    panic!("Branch position must point to a ReasoningOrToolCall content");
                };
                assert!(
                    position.offset > 0 && position.offset < original_tokens.tokens.len(),
                    "Branch position offset must be > 0 and < the length of the content tokens"
                );
                let first_half_tokens = original_tokens.tokens[..position.offset].to_vec();
                let first_half_logprobs = original_tokens.logprobs[..position.offset].to_vec();
                let second_half_tokens = original_tokens.tokens[position.offset..].to_vec();
                first_half_content_array.push(SegmentContent::ReasoningOrToolCall(
                    TokenArrayWithLogprob {
                        tokens: first_half_tokens.clone(),
                        decoded_string: <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(
                            &first_half_tokens,
                        ),
                        logprobs: first_half_logprobs, // we can fill in the logprobs later if needed
                    },
                ));
                second_half_content_array.insert(
                    0,
                    SegmentContent::ReasoningOrToolCall(TokenArrayWithLogprob {
                        tokens: second_half_tokens.clone(),
                        decoded_string: <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(
                            &second_half_tokens,
                        ),
                        logprobs: vec![], // we can fill in the logprobs later if needed
                    }),
                );
                let first_half_segment = Segment {
                    segment_id: new_first_half_id,
                    content: first_half_content_array,
                    child_ids: vec![new_second_half_id],
                    parent_id: target_segment.parent_id,
                };
                let second_half_segment = Segment {
                    segment_id: new_second_half_id,
                    content: second_half_content_array,
                    child_ids: target_segment.child_ids.clone(),
                    parent_id: Some(new_first_half_id),
                };
                self.segments.insert(new_first_half_id, first_half_segment);
                self.segments
                    .insert(new_second_half_id, second_half_segment);
                // check parent segment
                if let Some(parent_id) = target_segment.parent_id {
                    let parent_segment = self
                        .segments
                        .get_mut(&parent_id)
                        .expect("Parent segment must exist");
                    for child_id in &mut parent_segment.child_ids {
                        if *child_id == target_segment_id {
                            *child_id = new_first_half_id;
                            break;
                        }
                    }
                }
                // check child segments
                for child_id in &target_segment.child_ids {
                    let child_segment = self
                        .segments
                        .get_mut(child_id)
                        .expect("Child segment must exist");
                    child_segment.parent_id = Some(new_second_half_id);
                }
                // always check root
                for root_id in &mut self.root_segment_ids {
                    if *root_id == target_segment_id {
                        assert_eq!(
                            target_segment.parent_id, None,
                            "Root segment must not have a parent"
                        );
                        *root_id = new_first_half_id;
                        break;
                    }
                }
                // always checks leaf
                if let Some(judgment) = self.leaf_segment_judgments.remove(&target_segment_id) {
                    self.leaf_segment_judgments
                        .insert(new_second_half_id, judgment);
                }
                // after creating the branch point, we move to it and rollout until finding the answer
                self.focused_segment_id = Some(new_second_half_id);
                // update status
                self.status = DirectTreeStatus::CreatingBranchSegment;
            }
            DirectTreeAction::MoveToBranchPoint { target_segment_id } => {
                // this action does not change the tree structure, it only indicates that we are currently at a certain branch point
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingOrChoosingBranchPoint
                ));
                self.focused_segment_id = Some(target_segment_id);
                // update status
                self.status = DirectTreeStatus::CreatingBranchSegment;
            }
            DirectTreeAction::CreateAndFocusBranchSegment { content } => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingBranchSegment
                ));
                // this action adds a new segment as a child of the current branch point
                let Some(parent_id) = self.focused_segment_id else {
                    panic!("Must have a focused segment to add a branch segment");
                };
                // the focused segment is going to be the parent of the new segment
                let new_segment_id = self.next_segment_id;
                self.next_segment_id += 1;
                let new_segment = Segment {
                    segment_id: new_segment_id,
                    content,
                    child_ids: vec![],
                    parent_id: Some(parent_id),
                };
                self.segments.insert(new_segment_id, new_segment);
                if let Some(parent_segment) = self.segments.get_mut(&parent_id) {
                    parent_segment.child_ids.push(new_segment_id);
                }
                self.focused_segment_id = Some(new_segment_id);
                // update status
                self.status = DirectTreeStatus::JudgingBranchSegment;
            }
            DirectTreeAction::JudgeFocusedSegmentCorrectness {
                correctness_judgment,
            } => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::JudgingBranchSegment
                ));
                let Some(focused_segment_id) = self.focused_segment_id else {
                    panic!("Must have a focused segment to judge trajectory correctness");
                };
                assert!(
                    !self
                        .leaf_segment_judgments
                        .contains_key(&focused_segment_id),
                    "Focused segment must be a leaf segment that has not been judged before"
                );
                self.leaf_segment_judgments
                    .insert(focused_segment_id, correctness_judgment);
                // update status
                if self.segments.len() >= self.max_num_total_trajectories {
                    self.status = DirectTreeStatus::Complete;
                    self.completed = true;
                } else {
                    self.status = DirectTreeStatus::CreatingOrChoosingBranchPoint;
                }
            }
        }
    }
}

// it has interleaved reasoning and tool response
// we can branch on the reasoning part, but not on the tool response part
// tool response should not be counted towards the segment length
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment {
    pub segment_id: usize,
    pub content: Vec<SegmentContent>,
    pub child_ids: Vec<usize>,
    pub parent_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SegmentContent {
    Prompt(TokenArray),
    ReasoningOrToolCall(TokenArrayWithLogprob),
    ToolResponse(TokenArray),
}

// initially we need to finish 4 full trajectory rollouts.
// we can choose which trajectory to first branch on?

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirectTreeActionEntry {
    pub flat_id: usize,
    pub dataset_name: String,
    pub question_id: usize,
    pub action: DirectTreeAction,
}
