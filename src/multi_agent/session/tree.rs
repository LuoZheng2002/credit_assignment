use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::multi_agent::generate_rollout_answers::{CountRatio, StepQualityRatio};

use super::actions::{RolloutAction, ToolResponse, TrajectoryActionLog};
use super::constants::{
    CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE, IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE,
    REPETITION_ABORT_MESSAGE,
};
use super::types::{NextStepDecision, StepQuality, VerifierAndModeSummary};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub verifier_and_mode_summary: Option<VerifierAndModeSummary>,
    pub step_finalized: bool,
    pub step_quality: Option<StepQuality>,
    pub action_log: Vec<RolloutAction>,
}

impl Step {
    pub fn new() -> Self {
        Self {
            verifier_and_mode_summary: None,
            step_finalized: false,
            step_quality: None,
            action_log: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub node_id: usize,
    pub step: Step,
    pub verifier_on_child_id: Option<usize>,
    pub verifier_off_child_id: Option<usize>,
    pub parent_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessJudgment {
    pub model_answer: String,
    pub correct_answer: String,
    pub is_correct: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeMasterStatus {
    WorkingOnTrajectory,
    DeterminingBranchingNode,
}

// only working on one node at a time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub question_id: usize,
    pub question: String,
    pub nodes: Vec<Node>,
    pub root_node_id: Option<usize>,
    pub current_node_id: Option<usize>,
    pub leaf_node_ids: Vec<usize>,
    pub leaf_node_judgments: BTreeMap<usize, CorrectnessJudgment>,
    pub correctness_ratio: CountRatio,
    pub tool_wait_violations: usize,
    pub next_node_id: usize,
    pub tree_master_status: TreeMasterStatus,
}

// the following is the signature of each entry in a jsonl log file for reconstructing current tree progress when the program exits abruptly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeUpdateEvent {
    CreateNode {
        question_id: usize,
        node_id: usize,
        parent_id: Option<usize>,
        verifier_on: Option<bool>,
    },
    SetCurrentNode {
        question_id: usize,
        node_id: usize,
    },
    AddAction {
        question_id: usize,
        action: RolloutAction,
    },
    RegisterLeaf {
        question_id: usize,
        node_id: usize,
    },
    JudgeLeafCorrectness {
        question_id: usize,
        node_id: usize,
        correctness_judgment: CorrectnessJudgment,
    },
    ToolWaitViolation {
        question_id: usize,
    },
}

impl TreeUpdateEvent {
    pub fn question_id(&self) -> usize {
        match self {
            TreeUpdateEvent::CreateNode { question_id, .. } => *question_id,
            TreeUpdateEvent::SetCurrentNode { question_id, .. } => *question_id,
            TreeUpdateEvent::AddAction { question_id, .. } => *question_id,
            TreeUpdateEvent::RegisterLeaf { question_id, .. } => *question_id,
            TreeUpdateEvent::JudgeLeafCorrectness { question_id, .. } => *question_id,
            TreeUpdateEvent::ToolWaitViolation { question_id } => *question_id,
        }
    }
}

// we need a status after a trajectory is finished to randomly sample a node position for branching
// TrajectoryState is used for indicating the current status and what action should be generated in rollout.rs
// Eventually we apply the action to the Tree, and then we construct a TrajectoryState from the Tree for the current status and determine the next action to generate.
impl Tree {
    pub fn new(question_id: usize, question: String) -> Self {
        Self {
            question_id,
            question,
            nodes: Vec::new(),
            root_node_id: None,
            current_node_id: None,
            leaf_node_ids: Vec::new(),
            leaf_node_judgments: BTreeMap::new(),
            correctness_ratio: CountRatio {
                numerator: 0,
                denominator: 0,
            },
            tool_wait_violations: 0,
            next_node_id: 0,
            tree_master_status: TreeMasterStatus::WorkingOnTrajectory,
        }
    }

    pub fn append_action_to_current_node(&mut self, action: RolloutAction) {
        let current_node_id = self
            .current_node_id
            .expect("AddAction requires current_node_id to be set");
        let node = &mut self.nodes[current_node_id];
        let step = &mut node.step;
        assert!(
            !step.step_finalized,
            "Cannot append action to finalized step"
        );
        if let RolloutAction::PlannerDecideNextStep(mode) = &action {
            assert!(
                step.verifier_and_mode_summary.is_none(),
                "verifier_and_mode_summary should be set at most once per step"
            );
            let verifier_comment_in_current_step = step
                .action_log
                .iter()
                .find_map(|existing_action| match existing_action {
                    RolloutAction::VerifierComment(comment) => Some(comment.clone()),
                    _ => None,
                });
            step.verifier_and_mode_summary = Some(match (verifier_comment_in_current_step, mode) {
                (None, _) => VerifierAndModeSummary::VerifierOff,
                (Some(_), NextStepDecision::Continue) => VerifierAndModeSummary::VerifierOn,
                (Some(_), NextStepDecision::OverwriteLastStep(_)) => {
                    VerifierAndModeSummary::VerifierOnAndOverwriteLastStep
                }
                (Some(_), NextStepDecision::ChangePlan(_)) => {
                    VerifierAndModeSummary::VerifierOnAndChangePlan
                }
            });
        }
        if let RolloutAction::PlannerUpdatePlan(_) = &action {
            step.step_finalized = true;
        }
        if let RolloutAction::CompactorCompactStep { step_quality, .. } = &action {
            assert!(
                step.step_quality.is_none(),
                "PlannerCompactStep should set step_quality at most once per step"
            );
            step.step_quality = step_quality.clone();
        }
        if let RolloutAction::ToolCallResponse(ToolResponse::Intervention(content)) = &action {
            if content == CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE
                || content == IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE
                || content == REPETITION_ABORT_MESSAGE
            {
                step.step_finalized = true;
                assert!(
                    step.step_quality.is_none(),
                    "Terminal intervention should set step_quality at most once"
                );
                step.step_quality = Some(StepQuality::FailedAndAborted);
            }
        }
        step.action_log.push(action);
    }

    pub fn set_current_node_by_id(&mut self, node_id: usize) {
        let node = self
            .find_node_by_id(node_id)
            .expect("SetCurrentNode node_id must exist in tree");
        self.current_node_id = Some(node.node_id);
    }

    pub fn apply_event(&mut self, event: TreeUpdateEvent) {
        match event {
            TreeUpdateEvent::CreateNode {
                node_id,
                parent_id,
                verifier_on,
                ..
            } => {
                match parent_id {
                    None => {
                        assert_eq!(
                            verifier_on, None,
                            "Root CreateNode requires verifier_on to be None"
                        );
                        assert_eq!(node_id, 0, "Root node id must be 0");
                        assert!(self.nodes.is_empty(), "Root CreateNode must be first node event");
                        assert_eq!(
                            self.root_node_id, None,
                            "Root CreateNode requires empty root_node_id"
                        );
                        assert_eq!(
                            self.current_node_id, None,
                            "Root CreateNode requires empty current_node_id before explicit SetCurrentNode"
                        );
                        let root = Node {
                            node_id,
                            step: Step::new(),
                            verifier_on_child_id: None,
                            verifier_off_child_id: None,
                            parent_id: None,
                        };
                        self.nodes.push(root);
                        self.root_node_id = Some(node_id);
                    }
                    Some(parent_id) => {
                        let verifier_on = verifier_on
                            .expect("Non-root CreateNode requires verifier_on to be set");
                        assert!(
                            self.find_node_by_id(node_id).is_none(),
                            "CreateNode node_id must be unique"
                        );
                        let parent = self
                            .find_node_by_id(parent_id)
                            .expect("CreateNode parent_id must exist");
                        let parent_node_id = parent.node_id;
                        if verifier_on {
                            assert!(
                                self.nodes[parent_node_id].verifier_on_child_id.is_none(),
                                "CreateNode parent already has verifier_on child"
                            );
                        } else {
                            assert!(
                                self.nodes[parent_node_id].verifier_off_child_id.is_none(),
                                "CreateNode parent already has verifier_off child"
                            );
                        }
                        let child = Node {
                            node_id,
                            step: Step::new(),
                            verifier_on_child_id: None,
                            verifier_off_child_id: None,
                            parent_id: Some(parent_node_id),
                        };
                        self.nodes.push(child);
                        if verifier_on {
                            self.nodes[parent_node_id].verifier_on_child_id = Some(node_id);
                        } else {
                            self.nodes[parent_node_id].verifier_off_child_id = Some(node_id);
                        }
                    }
                }
                assert!(
                    node_id <= self.next_node_id,
                    "CreateNode node_id must not skip next_node_id"
                );
                if node_id == self.next_node_id {
                    self.next_node_id += 1;
                }
            }
            TreeUpdateEvent::SetCurrentNode { node_id, .. } => {
                self.set_current_node_by_id(node_id);
            }
            TreeUpdateEvent::AddAction { action, .. } => {
                self.append_action_to_current_node(action);
            }
            TreeUpdateEvent::RegisterLeaf { node_id, .. } => {
                let node = self
                    .find_node_by_id(node_id)
                    .expect("RegisterLeaf node_id must exist in tree");
                assert_eq!(
                    node.node_id, node_id,
                    "RegisterLeaf node index must equal node_id"
                );
                assert!(
                    !self.leaf_node_ids.contains(&node_id),
                    "RegisterLeaf should not register duplicate leaf node"
                );
                self.leaf_node_ids.push(node_id);
            }
            TreeUpdateEvent::JudgeLeafCorrectness {
                node_id,
                correctness_judgment,
                ..
            } => {
                let node = self
                    .find_node_by_id(node_id)
                    .expect("JudgeLeafCorrectness node_id must exist in tree");
                assert_eq!(
                    node.node_id, node_id,
                    "JudgeLeafCorrectness node index must equal node_id"
                );
                assert!(
                    self.leaf_node_ids.contains(&node_id),
                    "JudgeLeafCorrectness requires node_id to be a registered leaf"
                );
                assert!(
                    !self.leaf_node_judgments.contains_key(&node_id),
                    "JudgeLeafCorrectness should not overwrite an existing leaf judgment"
                );
                self.leaf_node_judgments.insert(node_id, correctness_judgment);
                let num_judged_leaves = self.leaf_node_judgments.len();
                assert!(
                    num_judged_leaves > 0,
                    "Leaf judgment map must be non-empty after inserting a leaf judgment"
                );
                let num_correct = self
                    .leaf_node_judgments
                    .values()
                    .filter(|judgment| judgment.is_correct)
                    .count();
                self.correctness_ratio = CountRatio {
                    numerator: num_correct,
                    denominator: num_judged_leaves,
                };
            }
            TreeUpdateEvent::ToolWaitViolation { .. } => {
                self.tool_wait_violations += 1;
            }
        }
    }

    fn find_node_by_id(&self, node_id: usize) -> Option<&Node> {
        let node = self.nodes.get(node_id)?;
        assert_eq!(node.node_id, node_id, "Node index must equal node_id");
        Some(node)
    }

    pub fn to_trajectory_log_on_current_path(&self) -> TrajectoryActionLog {
        let mut path_ids = Vec::new();
        let mut cursor = self.current_node_id;
        while let Some(node_id) = cursor {
            let node = self
                .find_node_by_id(node_id)
                .expect("Current-path node_id must exist in tree");
            path_ids.push(node_id);
            cursor = node.parent_id;
        }
        path_ids.reverse();

        let mut actions = Vec::new();
        for node_id in path_ids {
            let node = self
                .find_node_by_id(node_id)
                .expect("Current-path node_id must exist while collecting actions");
            actions.extend(node.step.action_log.iter().cloned());
        }
        TrajectoryActionLog(actions)
    }

    pub fn get_step_quality_ratio(&self) -> StepQualityRatio {
        let mut tool_true_count = 0usize;
        let mut complete_true_count = 0usize;
        let mut focused_true_count = 0usize;
        let mut total_count = 0usize;
        for node in &self.nodes {
            if let Some(step_quality) = &node.step.step_quality {
                total_count += 1;
                if let StepQuality::ProperlyEnded {
                    tool,
                    complete,
                    focused,
                } = step_quality
                {
                    if *tool {
                        tool_true_count += 1;
                    }
                    if *complete {
                        complete_true_count += 1;
                    }
                    if *focused {
                        focused_true_count += 1;
                    }
                }
            }
        }
        StepQualityRatio {
            tool_numerator: tool_true_count,
            tool_denominator: total_count,
            complete_numerator: complete_true_count,
            complete_denominator: total_count,
            focused_numerator: focused_true_count,
            focused_denominator: total_count,
        }
    }

    pub fn get_failed_and_aborted_ratio(&self) -> CountRatio {
        let mut failed_and_aborted_count = 0usize;
        for node in &self.nodes {
            if matches!(node.step.step_quality, Some(StepQuality::FailedAndAborted)) {
                failed_and_aborted_count += 1;
            }
        }
        CountRatio {
            numerator: failed_and_aborted_count,
            denominator: self.nodes.len(),
        }
    }
}
