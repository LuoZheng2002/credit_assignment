use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::multi_agent::generate_rollout_answers::{CountRatio, StepQualityRatio};
use crate::multi_agent::session::types::FinalAnswer;

use super::actions::{RolloutAction, TrajectoryActionLog};

use super::types::{NextStepDecision, StepQuality, VerifierAndModeSummary};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    // pub verifier_and_mode_summary: Option<VerifierAndModeSummary>,
    // pub step_finalized: bool,
    // pub step_quality: Option<StepQuality>,
    pub action_log: Vec<RolloutAction>,
}

impl Step {
    pub fn get_step_quality(&self) -> Option<StepQuality> {
        let step_quality = self
            .action_log
            .iter()
            .find_map(|action| match action {
                RolloutAction::CompactorCompactStep { step_quality, .. } => {
                    Some(step_quality.clone())
                }
                _ => None,
            })
            .expect("When getting step quality, the action log must have a compact step action");
        step_quality
    }
    pub fn verifier_and_mode_summary(&self) -> VerifierAndModeSummary {
        let chosen_mode = self
            .action_log
            .iter()
            .find_map(|action| match action {
                RolloutAction::PlannerDecideNextStep(mode) => Some(mode.clone()),
                _ => None,
            })
            .expect("When getting verifier and mode summary, the action log must have a PlannerDecideNextStep action");
        let verifier_comment = self
            .action_log
            .iter()
            .find_map(|action| match action {
                RolloutAction::VerifierComment(comment) => Some(comment.clone()),
                _ => None,
            })
            .expect("When getting verifier and mode summary, the action log must have a VerifierComment action");
        match (verifier_comment, chosen_mode) {
            (None, _) => VerifierAndModeSummary::VerifierOff,
            (Some(_), NextStepDecision::Continue) => VerifierAndModeSummary::VerifierOn,
            (Some(_), NextStepDecision::OverwriteLastStep(_)) => {
                VerifierAndModeSummary::VerifierOnAndOverwriteLastStep
            }
            (Some(_), NextStepDecision::ChangePlan(_)) => {
                VerifierAndModeSummary::VerifierOnAndChangePlan
            }
        }
    }
    pub fn step_finalized(&self) -> bool {
        self.action_log
            .iter()
            .find(|action| matches!(action, RolloutAction::StartNewStep))
            .is_some()
    }
}

// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub enum Step {
//     // pub verifier_and_mode_summary: Option<VerifierAndModeSummary>,
//     // pub step_finalized: bool,
//     // pub step_quality: Option<StepQuality>,
//     // pub action_log: Vec<RolloutAction>,

// }

impl Step {
    pub fn new() -> Self {
        Self {
            // verifier_and_mode_summary: None,
            // step_finalized: false,
            // step_quality: None,
            action_log: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub node_id: usize,
    pub step: Step,
    // pub verifier_on_child_id: Option<usize>,
    // pub verifier_off_child_id: Option<usize>,
    pub child_ids: [Option<usize>; 2],
    pub parent_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessJudgment {
    pub model_answer: FinalAnswer,
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
    pub reference_answer: String,
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
pub enum TreeAction {
    CreateNode {
        question_id: usize,
        node_id: usize,
        parent_id: Option<usize>,
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

// the whole tree can be:
//
// the trajectory state is based on the rollout actions but not tree events
// but we can reconstruct the rollout actions based on the current tree status
// so the rollout action is appended

impl TreeAction {
    pub fn question_id(&self) -> usize {
        match self {
            TreeAction::CreateNode { question_id, .. } => *question_id,
            TreeAction::SetCurrentNode { question_id, .. } => *question_id,
            TreeAction::AddAction { question_id, .. } => *question_id,
            TreeAction::RegisterLeaf { question_id, .. } => *question_id,
            TreeAction::JudgeLeafCorrectness { question_id, .. } => *question_id,
            TreeAction::ToolWaitViolation { question_id } => *question_id,
        }
    }
}

// we need a status after a trajectory is finished to randomly sample a node position for branching
// TrajectoryState is used for indicating the current status and what action should be generated in rollout.rs
// Eventually we apply the action to the Tree, and then we construct a TrajectoryState from the Tree for the current status and determine the next action to generate.
impl Tree {
    pub fn new(question_id: usize, question: String, reference_answer: String) -> Self {
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
            reference_answer,
        }
    }

    pub fn append_action_to_current_node(&mut self, action: RolloutAction) {
        let current_node = self
            .get_current_node_mut()
            .expect("AddAction requires current_node_id to be set and exist in nodes");
        let step = &mut current_node.step;
        assert!(
            !step.step_finalized(),
            "Cannot append action to finalized step"
        );
        step.action_log.push(action);
    }

    pub fn set_current_node_by_id(&mut self, node_id: usize) {
        assert!(
            node_id < self.nodes.len(),
            "SetCurrentNode node_id must exist in nodes"
        );
        self.current_node_id = Some(node_id);
    }

    pub fn apply_event(&mut self, event: TreeAction) {
        match event {
            TreeAction::CreateNode {
                node_id, parent_id, ..
            } => {
                if let Some(parent_id) = parent_id {
                    let child = Node {
                        node_id,
                        step: Step::new(),
                        child_ids: [None, None],
                        parent_id: Some(parent_id),
                    };
                    assert!(!self.has_id(node_id), "CreateNode node_id must be unique");
                    self.nodes.push(child);
                    let child_node_slot = self
                        .get_node_by_id_mut(parent_id)
                        .child_ids
                        .iter_mut()
                        .find(|child_id_option| child_id_option.is_none())
                        .expect("CreateNode parent should have an empty child slot");
                    *child_node_slot = Some(node_id);
                } else {
                    assert_eq!(node_id, 0, "Root node id must be 0");
                    assert!(
                        self.nodes.is_empty(),
                        "Root CreateNode must be first node event"
                    );
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
                        child_ids: [None, None],
                        parent_id: None,
                    };
                    self.nodes.push(root);
                    self.root_node_id = Some(node_id);
                }
                assert!(
                    node_id <= self.next_node_id,
                    "CreateNode node_id must not skip next_node_id"
                );
                if node_id == self.next_node_id {
                    self.next_node_id += 1;
                }
            }
            TreeAction::SetCurrentNode { node_id, .. } => {
                self.set_current_node_by_id(node_id);
            }
            TreeAction::AddAction { action, .. } => {
                self.append_action_to_current_node(action);
            }
            TreeAction::RegisterLeaf { node_id, .. } => {
                let node = self.get_node_by_id(node_id);
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
            TreeAction::JudgeLeafCorrectness {
                node_id,
                correctness_judgment,
                ..
            } => {
                let node = self.get_node_by_id(node_id);
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
                self.leaf_node_judgments
                    .insert(node_id, correctness_judgment);
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
            TreeAction::ToolWaitViolation { .. } => {
                self.tool_wait_violations += 1;
            }
        }
    }
    pub fn has_id(&self, node_id: usize) -> bool {
        self.nodes.iter().any(|node| node.node_id == node_id)
    }

    pub fn get_node_by_id(&self, node_id: usize) -> &Node {
        let node = self.nodes.get(node_id).expect("Node id must exist in tree");
        assert_eq!(node.node_id, node_id, "Node index must equal node_id");
        node
    }
    pub fn get_current_node(&self) -> Option<&Node> {
        let current_node_id = self.current_node_id?;
        Some(self.get_node_by_id(current_node_id))
    }
    pub fn get_current_node_mut(&mut self) -> Option<&mut Node> {
        let current_node_id = self.current_node_id?;
        Some(self.get_node_by_id_mut(current_node_id))
    }
    pub fn get_node_by_id_mut(&mut self, node_id: usize) -> &mut Node {
        let node = self
            .nodes
            .get_mut(node_id)
            .expect("Node id must exist in tree");
        assert_eq!(node.node_id, node_id, "Node index must equal node_id");
        node
    }

    pub fn to_trajectory_log_on_current_path(&self) -> TrajectoryActionLog {
        let mut path_ids = Vec::new();
        let mut cursor = self.current_node_id;
        while let Some(node_id) = cursor {
            let node = self.get_node_by_id(node_id);
            path_ids.push(node_id);
            cursor = node.parent_id;
        }
        path_ids.reverse();

        let mut actions = Vec::new();
        for node_id in path_ids {
            let node = self.get_node_by_id(node_id);
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
            if let Some(step_quality) = &node.step.get_step_quality() {
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
            if matches!(
                node.step.get_step_quality(),
                Some(StepQuality::FailedAndAborted)
            ) {
                failed_and_aborted_count += 1;
            }
        }
        CountRatio {
            numerator: failed_and_aborted_count,
            denominator: self.nodes.len(),
        }
    }
}
