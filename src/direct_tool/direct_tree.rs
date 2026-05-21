use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    agent::tree::CorrectnessJudgment,
    direct_tool::{
        direct_tree_action::DirectTreeAction, direct_tree_action_log::DirectTreeActionLog,
        direct_tree_status::DirectTreeStatus,
    },
    llm_model::{LlmModelMarker, MyTokenizer, TokenArrayWithLogprob, Top8Candidates},
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
    pub segments: BTreeMap<SegmentId, Segment>, // segment_id -> segment. A segment branched from the middle is destroyed and its id is not reused to avoid hiding sneaky bugs
    pub root_segment_ids: Vec<SegmentId>,
    pub leaf_segment_judgments: BTreeMap<SegmentId, CorrectnessJudgment>,
    pub next_segment_id: usize,
    pub next_segment_temperature: f32,
    pub focused_parent_segment_id: Option<SegmentId>, // the segment after which we create a new branch and rollout until finding the answer
    pub new_branch_start_token: Option<i32>, // the token id for the next branching point, which is determined when we create a branch and will be used in the rollout after branching to determine when to stop and judge the trajectory
    pub completed: bool,
    // hyperparameters
    pub num_trunks: usize,
    pub max_num_total_trajectories: usize,
    pub use_tool: bool,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<M>, // for tokenizer utility
}

pub const NUM_TRUNKS: usize = 4;

impl<M: LlmModelMarker> DirectTree<M> {
    // pub fn new(
    //     flat_id: usize,
    //     dataset_name: String,
    //     question_id: usize,
    //     question: String,
    //     correct_answer: String,
    //     num_trunks: usize,
    //     max_num_total_trajectories: usize,
    //     use_tool: bool,
    // ) -> Self {
    //     Self {
    //         flat_id,
    //         dataset_name,
    //         question_id,
    //         question,
    //         correct_answer,
    //         status: DirectTreeStatus::CreatingTrunkTrajectory,
    //         segments: BTreeMap::new(),
    //         root_segment_ids: vec![],
    //         leaf_segment_judgments: BTreeMap::new(),
    //         next_segment_id: 0,
    //         next_segment_temperature: 1.0,
    //         focused_parent_segment_id: None,
    //         new_branch_start_token: None,
    //         completed: false,
    //         num_trunks,
    //         max_num_total_trajectories,
    //         use_tool,
    //         _phantom: std::marker::PhantomData::<M>,
    //     }
    // }
    pub fn from_action_log(
        action_log: &DirectTreeActionLog,
        max_num_total_trajectories: usize,
        use_tool: bool,
    ) -> Self {
        let mut tree = Self {
            flat_id: action_log.question.flat_id,
            dataset_name: action_log.question.dataset_name.clone(),
            question_id: action_log.question.question_id,
            question: action_log.question.question.clone(),
            correct_answer: action_log.question.correct_answer.clone(),
            status: DirectTreeStatus::CreatingTrunkTrajectory, // this will be updated when applying actions
            segments: BTreeMap::new(),
            root_segment_ids: vec![],
            leaf_segment_judgments: BTreeMap::new(),
            next_segment_id: 0,
            next_segment_temperature: 1.0,
            focused_parent_segment_id: None,
            new_branch_start_token: None,
            completed: false,
            num_trunks: NUM_TRUNKS, // default value, will not affect the tree structure
            max_num_total_trajectories,
            use_tool,
            _phantom: std::marker::PhantomData::<M>,
        };
        for action in &action_log.actions {
            tree.apply_action(action.clone());
        }
        tree
    }
    fn apply_action(&mut self, action: DirectTreeAction) {
        match action {
            DirectTreeAction::CreateAndFocusTrunkTrajectory {
                content_array: content,
            } => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingTrunkTrajectory
                ));
                let segment_id = SegmentId(self.next_segment_id);
                self.next_segment_id += 1;
                self.segments.insert(
                    segment_id,
                    Segment {
                        segment_id,
                        content,
                        llm_temperature: self.next_segment_temperature,
                        child_ids: vec![],
                        parent_id: None,
                    },
                );
                self.root_segment_ids.push(segment_id);
                self.focused_parent_segment_id = Some(segment_id);
                // update status
                if self.root_segment_ids.len() >= self.num_trunks {
                    self.status = DirectTreeStatus::CreatingOrChoosingBranchPoint;
                } else {
                    self.status = DirectTreeStatus::CreatingTrunkTrajectory;
                }
            }
            DirectTreeAction::BranchFromSegment {
                position,
                new_branch_start_token,
            } => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingOrChoosingBranchPoint
                ));
                let new_first_half_id = SegmentId(self.next_segment_id);
                self.next_segment_id += 1;
                let new_second_half_id = SegmentId(self.next_segment_id);
                self.next_segment_id += 1;
                let target_segment = self
                    .segments
                    .remove(&position.segment_id)
                    .expect("Target segment id must exist");
                let target_content = &target_segment.content[position.content_index];
                let mut first_half_content_array =
                    target_segment.content[..position.content_index].to_vec();
                let mut second_half_content_array =
                    target_segment.content[position.content_index + 1..].to_vec();
                let SegmentContent::ReasoningOrToolCall(target_content_token_array) =
                    target_content
                else {
                    panic!("Branch position must point to a ReasoningOrToolCall content");
                };
                assert!(
                    position.offset > 0
                        && position.offset < target_content_token_array.tokens.len(),
                    "Branch position offset must be > 0 and < the length of the content tokens"
                );
                let first_half_tokens =
                    target_content_token_array.tokens[..position.offset].to_vec();
                let first_half_logprobs =
                    target_content_token_array.logprobs[..position.offset].to_vec();
                let second_half_tokens =
                    target_content_token_array.tokens[position.offset..].to_vec();
                let second_half_logprobs =
                    target_content_token_array.logprobs[position.offset..].to_vec();
                first_half_content_array.push(SegmentContent::ReasoningOrToolCall(
                    TokenArrayWithLogprob {
                        tokens: first_half_tokens.clone(),
                        decoded_string: <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(
                            &first_half_tokens,
                        ),
                        logprobs: first_half_logprobs,
                    },
                ));
                second_half_content_array.insert(
                    0,
                    SegmentContent::ReasoningOrToolCall(TokenArrayWithLogprob {
                        tokens: second_half_tokens.clone(),
                        decoded_string: <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(
                            &second_half_tokens,
                        ),
                        logprobs: second_half_logprobs,
                    }),
                );
                let first_half_segment = Segment {
                    segment_id: new_first_half_id,
                    content: first_half_content_array,
                    child_ids: vec![new_second_half_id],
                    parent_id: target_segment.parent_id,
                    llm_temperature: target_segment.llm_temperature,
                };
                let second_half_segment = Segment {
                    segment_id: new_second_half_id,
                    content: second_half_content_array,
                    child_ids: target_segment.child_ids.clone(),
                    parent_id: Some(new_first_half_id),
                    llm_temperature: target_segment.llm_temperature,
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
                        if *child_id == position.segment_id {
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
                    if *root_id == position.segment_id {
                        assert_eq!(
                            target_segment.parent_id, None,
                            "Root segment must not have a parent"
                        );
                        *root_id = new_first_half_id;
                        break;
                    }
                }
                // always checks leaf
                if let Some(judgment) = self.leaf_segment_judgments.remove(&position.segment_id) {
                    self.leaf_segment_judgments
                        .insert(new_second_half_id, judgment);
                }
                // after creating the branch point, we move to it and rollout until finding the answer
                // the new branch point is at the end of the first half segment
                self.focused_parent_segment_id = Some(new_first_half_id);
                self.new_branch_start_token = Some(new_branch_start_token);
                // update status
                self.status = DirectTreeStatus::CreatingBranchSegment;
            }
            DirectTreeAction::BranchFromNode {
                position,
                new_branch_start_token,
            } => {
                // this action does not change the tree structure, it only indicates that we are currently at a certain branch point
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingOrChoosingBranchPoint
                ));
                assert!(
                    position.content_index == 0 && position.offset == 0,
                    "Branch from node action must branch at the boundary between segments"
                );
                let parent_id = self
                    .segments
                    .get(&position.segment_id)
                    .expect("Target segment id must exist")
                    .parent_id
                    .expect("Target segment must have a parent segment to branch from a node");
                self.focused_parent_segment_id = Some(parent_id);
                self.new_branch_start_token = Some(new_branch_start_token);
                // update status
                self.status = DirectTreeStatus::CreatingBranchSegment;
            }
            DirectTreeAction::NoAvailableBranchPoint => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingOrChoosingBranchPoint
                ));
                // this action does not change the tree structure, it only indicates that we have found no valid branching point and should conclude the tree
                self.status = DirectTreeStatus::Complete;
                self.completed = true;
            }
            DirectTreeAction::CreateAndFocusBranchSegment { content } => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::CreatingBranchSegment
                ));
                // this action adds a new segment as a child of the current branch point
                let Some(parent_id) = self.focused_parent_segment_id else {
                    panic!("Must have a focused segment to add a branch segment");
                };
                // the focused segment is going to be the parent of the new segment
                let new_segment_id = SegmentId(self.next_segment_id);
                self.next_segment_id += 1;
                let new_segment = Segment {
                    segment_id: new_segment_id,
                    content,
                    child_ids: vec![],
                    parent_id: Some(parent_id),
                    llm_temperature: self.next_segment_temperature,
                };
                self.segments.insert(new_segment_id, new_segment);
                if let Some(parent_segment) = self.segments.get_mut(&parent_id) {
                    parent_segment.child_ids.push(new_segment_id);
                }
                self.focused_parent_segment_id = Some(new_segment_id);
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
                let Some(focused_segment_id) = self.focused_parent_segment_id else {
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

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentId(usize);

// it has interleaved reasoning and tool response
// we can branch on the reasoning part, but not on the tool response part
// tool response should not be counted towards the segment length
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment {
    pub segment_id: SegmentId,
    pub content: Vec<SegmentContent>,
    pub llm_temperature: f32,
    pub child_ids: Vec<SegmentId>,
    pub parent_id: Option<SegmentId>,
}
// #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
// pub struct ReasoningContentIndex(usize);
// pub struct ReasoningOnlySegmentView<'a> {
//     // pub reasoning_contents: Vec<TokenArrayWithLogprob>,

//     pub corresponding_segment: &'a Segment,
// }

pub struct ReasoningOnlyTokenView<'a> {
    pub flat_index: usize, // the index of the token in the flattened reasoning-only token sequence of the segment
    pub token: i32,
    pub logprobs: Top8Candidates,
    pub content_index_in_segment: usize, // the index of the content in the original segment content array that this token belongs to
    pub token_offset_in_content: usize,  // the offset of the token in the original content tokens
    pub corresponding_segment: &'a Segment,
}

impl Segment {
    pub fn reasoning_only_tokens<'a>(&'a self) -> Vec<ReasoningOnlyTokenView<'a>> {
        let mut views = vec![];
        let mut flat_index = 0;
        for (content_index, content) in self.content.iter().enumerate() {
            if let SegmentContent::ReasoningOrToolCall(tokens) = content {
                for (token_offset, (&token, logprobs)) in
                    tokens.tokens.iter().zip(tokens.logprobs.iter()).enumerate()
                {
                    views.push(ReasoningOnlyTokenView {
                        flat_index,
                        token,
                        logprobs: *logprobs,
                        content_index_in_segment: content_index,
                        token_offset_in_content: token_offset,
                        corresponding_segment: self,
                    });
                    flat_index += 1;
                }
            }
        }
        views
    }
    pub fn first_reasoning_token(&self) -> Option<i32> {
        for content in &self.content {
            if let SegmentContent::ReasoningOrToolCall(tokens) = content {
                if let Some(&first_token) = tokens.tokens.first() {
                    return Some(first_token);
                }
            }
        }
        None
    }
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
