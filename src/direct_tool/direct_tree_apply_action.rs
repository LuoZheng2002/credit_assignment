use research_utility::log_message::log_info;

use crate::{
    direct_tool::{
        direct_rollout_config::BranchingPolicy,
        direct_tree::{DirectTree, Segment, SegmentContent, SegmentId},
        direct_tree_action::{DirectTreeAction, TokenPositionInTree},
        direct_tree_spontaneous_branching::TokenPositionInSegment,
        direct_tree_status::{
            DirectTreeStatus, GuidedBranchingSubStatus, SpontaneousBranchingSubStatus,
            TrunkSubStatus,
        },
    },
    llm_model::{LlmModelMarker, TokenArrayWithLogprob},
};

impl<'a, M: LlmModelMarker> DirectTree<'a, M> {
    pub fn apply_action(&mut self, action: &DirectTreeAction<M>) {
        match action {
            DirectTreeAction::AppendSegmentContent(segment_content) => match &mut self.status {
                DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::CollectingSegmentContents {
                    cumulative_content_array,
                })
                | DirectTreeStatus::WorkingOnGuidedBranching(
                    GuidedBranchingSubStatus::CollectingSegmentContents {
                        cumulative_content_array,
                        ..
                    },
                )
                | DirectTreeStatus::WorkingOnSpontaneousBranching(
                    SpontaneousBranchingSubStatus::CollectingSegmentContents {
                        cumulative_content_array,
                    },
                ) => {
                    cumulative_content_array.push(segment_content.clone());
                }
                _ => unreachable!(),
            },
            DirectTreeAction::SubmitAnswer(final_answer) => {
                let final_answer = final_answer.clone();
                let new_status = match self.status.clone() {
                    DirectTreeStatus::WorkingOnTrunk(
                        TrunkSubStatus::CollectingSegmentContents {
                            cumulative_content_array,
                        },
                    ) => DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::JudgingSegment {
                        final_answer,
                        finalized_content_array: cumulative_content_array,
                    }),
                    DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::CollectingSegmentContents {
                            cumulative_content_array,
                            parent_segment_id,
                            new_branch_start_token: _,
                        },
                    ) => DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::JudgingSegment {
                            final_answer,
                            parent_segment_id,
                            finalized_content_array: cumulative_content_array,
                        },
                    ),
                    DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::CollectingSegmentContents {
                            cumulative_content_array,
                        },
                    ) => DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::DeterminingBranchingPoint {
                            finalized_content_array: cumulative_content_array,
                            final_answer,
                        },
                    ),
                    _ => unreachable!(),
                };
                self.status = new_status;
            }
            DirectTreeAction::BranchFromSegmentOrNodeGuided {
                position,
                new_branch_start_token,
                branch_from_node,
            } => {
                let position = position.clone();
                let new_branch_start_token = *new_branch_start_token;
                let branch_from_node = *branch_from_node;
                let new_status = match self.status.clone() {
                    DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::DeterminingBranchingPoint,
                    ) => DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::SplittingTargetSegment {
                            position,
                            branch_from_node,
                            new_branch_start_token,
                        },
                    ),
                    _ => unreachable!(),
                };
                self.status = new_status;
            }
            DirectTreeAction::BranchFromSegmentOrNodeSpontaneous {
                position,
                branch_from_node,
                position_in_segment,
            } => {
                let position = position.clone();
                let branch_from_node = *branch_from_node;
                let position_in_segment = position_in_segment.clone();
                let new_status = match self.status.clone() {
                    DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::DeterminingBranchingPoint {
                            finalized_content_array,
                            final_answer,
                        },
                    ) => DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::PrefixTrimmingNewSegment {
                            position,
                            position_in_segment,
                            finalized_content_array,
                            branch_from_node,
                            final_answer,
                        },
                    ),
                    _ => unreachable!(),
                };
                self.status = new_status;
            }
            DirectTreeAction::NoAvailableBranchPoint => {
                assert!(matches!(
                    self.status,
                    DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::DeterminingBranchingPoint
                    ) | DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::DeterminingBranchingPoint { .. }
                    )
                ));
                self.status = DirectTreeStatus::Complete;
            }
            DirectTreeAction::PrefixTrimNewSegment { trim_position } => {
                let new_status = match self.status.clone() {
                    DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::PrefixTrimmingNewSegment {
                            position,
                            position_in_segment: _,
                            finalized_content_array,
                            branch_from_node,
                            final_answer,
                        },
                    ) => {
                        let prefix_trimmed_content_array =
                            Self::prefix_trim_content_array(finalized_content_array, trim_position);
                        DirectTreeStatus::WorkingOnSpontaneousBranching(
                            SpontaneousBranchingSubStatus::SplittingTargetSegment {
                                position,
                                branch_from_node,
                                prefix_trimmed_content_array,
                                final_answer,
                            },
                        )
                    }
                    _ => unreachable!(),
                };
                self.status = new_status;
            }
            DirectTreeAction::SplitTreeSegment {
                position,
                branch_from_node,
            } => {
                let parent_segment_id = if !branch_from_node {
                    let SplitResult { new_first_half_id } = self.split_segment(position);
                    new_first_half_id
                } else {
                    assert!(
                        position.content_index == 0 && position.offset == 0,
                        "Branch from node action must branch at the boundary between segments"
                    );
                    let segment = self
                        .segments
                        .get(&position.segment_id)
                        .expect("Target segment id must exist in tree when splitting");
                    segment.parent_id.expect("Target segment must have a parent segment to branch from a node when splitting")
                };
                let new_status = match self.status.clone() {
                    DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::SplittingTargetSegment {
                            position: _,
                            branch_from_node: _,
                            new_branch_start_token,
                        },
                    ) => DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::CollectingSegmentContents {
                            cumulative_content_array: Vec::new(),
                            parent_segment_id,
                            new_branch_start_token,
                        },
                    ),
                    DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::SplittingTargetSegment {
                            position: _,
                            branch_from_node: _,
                            prefix_trimmed_content_array,
                            final_answer,
                        },
                    ) => DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::JudgingSegment {
                            final_answer,
                            parent_segment_id,
                            prefix_trimmed_content_array,
                        },
                    ),
                    _ => unreachable!(),
                };
                self.status = new_status;
            }
            DirectTreeAction::JudgeAnswer(correctness_judgment) => {
                let correctness_judgment = correctness_judgment.clone();
                let new_status = match self.status.clone() {
                    DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::JudgingSegment {
                        final_answer: _,
                        finalized_content_array,
                    }) => DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::AttachingToTree {
                        correctness_judgment,
                        parent_segment_id: self.root_segment_id.unwrap(),
                        finalized_content_array,
                    }),
                    DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::JudgingSegment {
                            final_answer: _,
                            parent_segment_id,
                            finalized_content_array,
                        },
                    ) => DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::AttachingToTree {
                            correctness_judgment,
                            parent_segment_id,
                            finalized_content_array,
                        },
                    ),
                    DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::JudgingSegment {
                            final_answer: _,
                            parent_segment_id,
                            prefix_trimmed_content_array,
                        },
                    ) => DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::AttachingToTree {
                            correctness_judgment,
                            parent_segment_id,
                            prefix_trimmed_content_array,
                        },
                    ),
                    _ => unreachable!(),
                };
                self.status = new_status;
            }
            DirectTreeAction::AttachSegmentToTree {
                parent_segment_id,
                finalized_content_array,
                correctness_judgment,
            } => {
                let new_segment_id = SegmentId(self.next_segment_id);
                self.next_segment_id += 1;
                let new_segment = Segment {
                    segment_id: new_segment_id,
                    content: finalized_content_array.clone(),
                    child_ids: vec![],
                    parent_id: Some(*parent_segment_id),
                };
                self.segments.insert(new_segment_id, new_segment);
                let Some(parent_segment) = self.segments.get_mut(&parent_segment_id) else {
                    panic!("Parent segment must exist");
                };
                parent_segment.child_ids.push(new_segment_id);
                assert!(!self.leaf_segment_judgments.contains_key(&new_segment_id));
                self.leaf_segment_judgments
                    .insert(new_segment_id, correctness_judgment.clone());
                match &self.status {
                    DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::AttachingToTree {
                        ..
                    }) => {
                        // we are still working on the trunk, so we add the new segment to trunk leaf segments
                        self.trunk_leaf_segments.insert(new_segment_id);
                    }
                    DirectTreeStatus::WorkingOnGuidedBranching(
                        GuidedBranchingSubStatus::AttachingToTree { .. },
                    )
                    | DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::AttachingToTree { .. },
                    ) => {
                        // do nothing
                    }
                    _ => unreachable!(),
                }
                self.status = self.determine_status_after_segment_attachment();
            }
        };
    }
    fn determine_status_after_segment_attachment(&self) -> DirectTreeStatus<M> {
        // we can choose to work on trunk, (guided branch or spontaneous branch), or conclude the tree
        if self.trunk_leaf_segments.len() < self.action_log.rollout_config.max_num_trunks {
            DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::CollectingSegmentContents {
                cumulative_content_array: vec![],
            })
        } else if self.leaf_segment_judgments.len()
            < self.action_log.rollout_config.max_num_total_trajectories
        {
            match self.action_log.rollout_config.branching_policy {
                BranchingPolicy::TreeMappoGuided => DirectTreeStatus::WorkingOnGuidedBranching(
                    GuidedBranchingSubStatus::DeterminingBranchingPoint,
                ),
                BranchingPolicy::TempoSpontaneous => {
                    DirectTreeStatus::WorkingOnSpontaneousBranching(
                        SpontaneousBranchingSubStatus::CollectingSegmentContents {
                            cumulative_content_array: Vec::new(),
                        },
                    )
                }
            }
        } else {
            DirectTreeStatus::Complete
        }
    }
    fn split_segment(&mut self, position: &TokenPositionInTree) -> SplitResult {
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
        let SegmentContent::ReasoningOrToolCall {
            tokens,
            complete: target_is_complete,
        } = target_content
        else {
            panic!("Branch position must point to a ReasoningOrToolCall content");
        };
        assert!(
            (position.offset > 0 || position.content_index > 0)
                && position.offset < tokens.tokens.len(),
            "Branch position offset must be > 0 and < the length of the content tokens. position.offset: {}, tokens length: {}",
            position.offset,
            tokens.tokens.len()
        );
        let first_half_tokens = tokens.tokens[..position.offset].to_vec();
        let first_half_logprobs = tokens.logprobs[..position.offset].to_vec();
        let second_half_tokens = tokens.tokens[position.offset..].to_vec();
        let second_half_logprobs = tokens.logprobs[position.offset..].to_vec();
        first_half_content_array.push(SegmentContent::ReasoningOrToolCall {
            tokens: TokenArrayWithLogprob::from_tokens_and_logprobs(
                first_half_tokens.clone(),
                first_half_logprobs,
            ),
            complete: false, // the first half is always incomplete
        });
        second_half_content_array.insert(
            0,
            SegmentContent::ReasoningOrToolCall {
                tokens: TokenArrayWithLogprob::from_tokens_and_logprobs(
                    second_half_tokens.clone(),
                    second_half_logprobs,
                ),
                complete: *target_is_complete, // the second half is complete if and only if the original content is complete
            },
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
        assert!(position.segment_id != self.root_segment_id.expect("Root id must exist"));
        // always checks leaf
        if let Some(judgment) = self.leaf_segment_judgments.remove(&position.segment_id) {
            self.leaf_segment_judgments
                .insert(new_second_half_id, judgment);
        }
        if let true = self.trunk_leaf_segments.remove(&position.segment_id) {
            self.trunk_leaf_segments.insert(new_second_half_id);
        }
        SplitResult { new_first_half_id }
    }
    fn prefix_trim_content_array(
        content_array: Vec<SegmentContent<M>>,
        trim_position: &TokenPositionInSegment,
    ) -> Vec<SegmentContent<M>> {
        assert!(
            trim_position.content_index < content_array.len(),
            "Trim content index out of bounds"
        );

        let mut prefix_trimmed = Vec::new();
        for (content_index, content) in content_array.into_iter().enumerate() {
            if content_index < trim_position.content_index {
                continue;
            }
            if content_index > trim_position.content_index {
                prefix_trimmed.push(content);
                continue;
            }

            let offset = trim_position.offset;
            match content {
                SegmentContent::Prompt(mut tokens) => {
                    assert!(offset <= tokens.tokens.len(), "Trim offset out of bounds");
                    if offset == tokens.tokens.len() {
                        continue;
                    }
                    tokens.tokens.drain(0..offset);
                    prefix_trimmed.push(SegmentContent::Prompt(tokens));
                }
                SegmentContent::ToolResponse(mut tokens) => {
                    assert!(offset <= tokens.tokens.len(), "Trim offset out of bounds");
                    if offset == tokens.tokens.len() {
                        continue;
                    }
                    tokens.tokens.drain(0..offset);
                    prefix_trimmed.push(SegmentContent::ToolResponse(tokens));
                }
                SegmentContent::ReasoningOrToolCall { tokens, complete } => {
                    assert!(offset <= tokens.tokens.len(), "Trim offset out of bounds");
                    if offset == tokens.tokens.len() {
                        continue;
                    }
                    let trimmed_tokens = tokens.tokens[offset..].to_vec();
                    let trimmed_logprobs = tokens.logprobs[offset..].to_vec();
                    prefix_trimmed.push(SegmentContent::ReasoningOrToolCall {
                        tokens: TokenArrayWithLogprob::from_tokens_and_logprobs(
                            trimmed_tokens,
                            trimmed_logprobs,
                        ),
                        complete,
                    });
                }
            }
        }

        assert!(
            !prefix_trimmed.is_empty(),
            "Prefix trimming should preserve at least one content"
        );
        prefix_trimmed
    }
}

struct SplitResult {
    new_first_half_id: SegmentId,
}
