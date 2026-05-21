use crate::{
    direct_tool::{
        direct_tree::{DirectTree, Segment, SegmentContent, SegmentId},
        direct_tree_action::DirectTreeAction,
        direct_tree_status::DirectTreeStatus,
    },
    llm_model::{LlmModelMarker, MyTokenizer, TokenArrayWithLogprob},
};

impl<M: LlmModelMarker> DirectTree<M> {
    pub fn apply_action(&mut self, action: DirectTreeAction) {
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
            DirectTreeAction::CreateAndFocusBranchSegment { contents: content } => {
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
