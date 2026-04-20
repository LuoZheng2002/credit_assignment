use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::deepmath::parse_answers::extract_boxed_content;
use crate::multi_agent::generate_rollout_answers::StepQualityAccuracy;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NextStepDecision {
    Continue,
    OverwriteLastStep(String),
    ChangePlan(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MakeOrChangePlan {
    MakePlan(String),
    ChangePlan {
        plan: String,
        prev_failed_reason: String,
    },
}

impl NextStepDecision {
    pub fn is_overwriting(&self) -> bool {
        match self {
            NextStepDecision::Continue => false,
            NextStepDecision::OverwriteLastStep(_) => true,
            NextStepDecision::ChangePlan(_) => panic!(
                "ChangePlan should not be a valid step mode when the planner has entered the working on step status"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletedStep {
    // pub step_mode: DisplayStepMode,
    pub current_step_mode: NextStepDecision,
    pub content_raw: String,
    pub content_compacted: String,
    pub step_quality: Option<StepQuality>,
    pub current_step_verifier_comment: Option<VerifierComment>,
}
impl CompletedStep {
    pub fn new(
        current_step_mode: NextStepDecision,
        content_raw: String,
        content_compacted: String,
        step_quality: Option<StepQuality>,
        current_step_verifier_comment: Option<VerifierComment>,
    ) -> Self {
        Self {
            current_step_mode,
            content_raw,
            content_compacted,
            step_quality,
            current_step_verifier_comment,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepQuality {
    pub tool: bool,
    pub complete: bool,
    pub focused: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierComment {
    pub comment: String,
    pub overwrite: bool,
    pub change_plan: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolResponse {
    PythonSuccess(String),
    PythonError(String),
    Intervention(String),
}

impl ToolResponse {
    pub fn to_raw_content(&self) -> String {
        match self {
            ToolResponse::PythonSuccess(output) => {
                format!("<tool_response>{}</tool_response>", output)
            }
            ToolResponse::PythonError(error) => {
                format!("<tool_response>Python error: {}</tool_response>", error)
            }
            ToolResponse::Intervention(content) => content.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutAction {
    PlannerMakeOrChangePlan(Option<MakeOrChangePlan>), // this action is always part of the routine; None means no new plan is made
    PlannerDecideNextStep(NextStepDecision),
    PlannerReasoning(String),
    PlannerToolCall(String), // with <tool_call> ... </tool_call> wrapper
    ToolCallResponse(ToolResponse),
    PlannerEndStep,
    // PlannerCompactStep(String, Option<StepQuality>),
    PlannerCompactStep {
        summary: String,
        step_quality: Option<StepQuality>,
    },
    PlannerUpdatePlan(String),
    VerifierComment(Option<VerifierComment>),
}

impl RolloutAction {
    pub fn to_pretty_string(&self) -> String {
        match self {
            RolloutAction::PlannerMakeOrChangePlan(plan) => {
                format!("[PlannerMakeOrChangePlan]:\n{:?}", plan)
            }
            RolloutAction::PlannerDecideNextStep(mode) => {
                format!("[PlannerChooseMode]: {:?}", mode)
            }
            RolloutAction::PlannerReasoning(reasoning) => {
                format!("[PlannerReasoning]:\n{}", reasoning)
            }
            RolloutAction::PlannerToolCall(tool_call) => {
                format!("[PlannerToolCall]:\n{}", tool_call)
            }
            RolloutAction::PlannerEndStep => "[PlannerEndStep]".to_string(),
            RolloutAction::PlannerCompactStep {
                summary,
                step_quality,
            } => {
                format!(
                    "[PlannerCompactStep]:\n{}\n[StepQuality]: {:?}",
                    summary, step_quality
                )
            }
            RolloutAction::PlannerUpdatePlan(updated_plan) => {
                format!("[PlannerUpdatePlan]:\n{}", updated_plan)
            }
            RolloutAction::ToolCallResponse(tool_response) => {
                format!("[ToolCallResponse]:\n{}", tool_response.to_raw_content())
            }
            RolloutAction::VerifierComment(comment) => {
                format!("[VerifierComment]:\n{:?}", comment)
            }
        }
    }
    pub fn to_concise_string(&self) -> String {
        match self {
            RolloutAction::PlannerMakeOrChangePlan(_plan) => {
                format!("[PlannerMakeOrChangePlan]")
            }
            RolloutAction::PlannerDecideNextStep(mode) => {
                format!("[PlannerChooseMode]: {:?}", mode)
            }
            RolloutAction::PlannerReasoning(_reasoning) => "[PlannerReasoning]".to_string(),
            RolloutAction::PlannerToolCall(_tool_call) => "[PlannerToolCall]".to_string(),
            RolloutAction::PlannerEndStep => "[PlannerEndStep]".to_string(),
            RolloutAction::PlannerCompactStep { step_quality, .. } => {
                format!("[PlannerCompactStep]\n[StepQuality]: {:?}", step_quality)
            }
            RolloutAction::PlannerUpdatePlan(_updated_plan) => "[PlannerUpdatePlan]".to_string(),
            RolloutAction::ToolCallResponse(_tool_response) => "[ToolCallResponse]".to_string(),
            RolloutAction::VerifierComment(comment) => {
                format!("[VerifierComment]\n{:?}", comment)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryActionLog(pub Vec<RolloutAction>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryStatus {
    PlannerMakingOrChangingPlan, // this status always pushes PlannerMakeOrChangePlan(Some(...))
    PlannerKeepingCurrentPlan,   // this status always pushes PlannerMakeOrChangePlan(None)
    PlannerChoosingMode,
    PlannerWorkingOnStep,
    PlannerCompactingStep,
    PlannerUpdatingPlan,
    VerifierCommenting,
}

#[derive(Debug, Clone)]
pub struct FailedAttempt {
    pub plan: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct TrajectoryState<'a> {
    pub source_tree: &'a Tree,
    pub question: String,
    pub prev_steps: Vec<CompletedStep>,
    pub current_step_content_raw: String,
    pub current_step_content_compacted: Option<String>,
    pub current_step_quality: Option<StepQuality>,
    pub current_plan: Option<String>,
    pub status: TrajectoryStatus,
    pub planner_chosen_mode: Option<NextStepDecision>,
    pub final_answer: Option<String>,
    pub failed_attempts: Vec<FailedAttempt>,
    pub num_plan_changes: usize,
    pub step_overwrite_streak: usize,
    pub total_step_overwrites: usize,
    pub current_step_num_actions: usize,
    pub current_step_last_python_error: Option<String>,
    pub total_actions: usize,
    pub total_actual_steps: usize,
    pub step_quality_tool_true_count: usize,
    pub step_quality_tool_total_count: usize,
    pub step_quality_complete_true_count: usize,
    pub step_quality_complete_total_count: usize,
    pub step_quality_focused_true_count: usize,
    pub step_quality_focused_total_count: usize,
    pub should_end_session: bool,
}

pub const FORCED_END_MESSAGE: &str =
    "The model does not manage to provide a final answer within allowed number of turns.";
pub const CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE: &str =
    "<error>Model context length exceeded, aborting.</error>";
pub const IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE: &str =
    "<error>Identical python tool error detected. Aborting current incomplete step.</error>";
pub const REPETITION_ABORT_MESSAGE: &str =
    "<error>Repeated contents detected. This step is forced to abort without completion.</error>";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifierAndModeSummary {
    VerifierOff,
    VerifierOn,
    VerifierOnAndOverwriteLastStep,
    VerifierOnAndChangePlan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub verifier_and_mode_summary: Option<VerifierAndModeSummary>, // Some after verifier comment and next step decision
    pub step_finalized: bool, // initialized as false, true once PlannerUpdatePlan or terminal intervention has been emitted
    pub action_log: Vec<RolloutAction>, // starting from verifier comment, and ending with planner end step or force end step
}

impl Step {
    pub fn new() -> Self {
        Self {
            verifier_and_mode_summary: None,
            step_finalized: false,
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

// only working on one node at a time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub question_id: usize,
    pub question: String,
    pub nodes: Vec<Node>,
    pub root_node_id: Option<usize>,
    pub current_node_id: Option<usize>,
    pub leaf_node_ids: Vec<usize>, // this is only for trajectories that have reached the final answer or is forced to end
    pub leaf_node_judgments: BTreeMap<usize, CorrectnessJudgment>,
    pub accuracy: f64,
    pub next_node_id: usize,
    pub tree_master_status: TreeMasterStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeMasterStatus {
    WorkingOnTrajectory,
    DeterminingBranchingNode,
}

// the following is the signature of each entry in a jsonl log file for reconstructing current tree progress when the program exits abruptly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeUpdateEvent {
    CreateNode {
        question_id: usize,
        node_id: usize,
        parent_id: Option<usize>, // None for root node
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
}

// we need a status after a trajectory is finished to randomly sample a node position for branching
// TrajectoryState is used for indicating the current status and what action should be generated in rollout.rs
// Eventually we apply the action to the Tree, and then we construct a TrajectoryState from the Tree for the current status and determine the next action to generate.

impl TreeUpdateEvent {
    pub fn question_id(&self) -> usize {
        match self {
            TreeUpdateEvent::CreateNode { question_id, .. } => *question_id,
            TreeUpdateEvent::SetCurrentNode { question_id, .. } => *question_id,
            TreeUpdateEvent::AddAction { question_id, .. } => *question_id,
            TreeUpdateEvent::RegisterLeaf { question_id, .. } => *question_id,
            TreeUpdateEvent::JudgeLeafCorrectness { question_id, .. } => *question_id,
        }
    }
}

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
            accuracy: 0.0,
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
        if let RolloutAction::ToolCallResponse(ToolResponse::Intervention(content)) = &action {
            if content == CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE
                || content == IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE
                || content == REPETITION_ABORT_MESSAGE
            {
                step.step_finalized = true;
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
                            verifier_on,
                            None,
                            "Root CreateNode requires verifier_on to be None"
                        );
                        assert_eq!(node_id, 0, "Root node id must be 0");
                        assert!(self.nodes.is_empty(), "Root CreateNode must be first node event");
                        assert_eq!(
                            self.root_node_id,
                            None,
                            "Root CreateNode requires empty root_node_id"
                        );
                        assert_eq!(
                            self.current_node_id,
                            None,
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
                self.accuracy = num_correct as f64 / num_judged_leaves as f64;
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
}

pub const MAX_PLAN_CHANGES: usize = 2;
pub const MAX_STEP_OVERWRITE_STREAK: usize = 2;
pub const MAX_TOTAL_STEP_OVERWRITES: usize = 6;
pub const MAX_ACTIONS_PER_STEP: usize = 30;
impl<'a> TrajectoryState<'a> {
    fn new(question: String, source_tree: &'a Tree) -> Self {
        Self {
            source_tree,
            question,
            prev_steps: Vec::new(),
            current_step_content_raw: String::new(),
            current_step_quality: None,
            num_plan_changes: 0,
            current_step_num_actions: 0,
            current_step_last_python_error: None,
            step_overwrite_streak: 0,
            total_step_overwrites: 0,
            current_step_content_compacted: None,
            current_plan: None,
            status: TrajectoryStatus::VerifierCommenting,
            planner_chosen_mode: None,
            final_answer: None,
            failed_attempts: Vec::new(),
            total_actions: 0,
            step_quality_tool_true_count: 0,
            step_quality_tool_total_count: 0,
            step_quality_complete_true_count: 0,
            step_quality_complete_total_count: 0,
            step_quality_focused_true_count: 0,
            step_quality_focused_total_count: 0,
            total_actual_steps: 0,
            should_end_session: false,
        }
    }
    pub fn from_session_log(
        question: String,
        session_log: TrajectoryActionLog,
        source_tree: &'a Tree,
    ) -> Self {
        let mut session_state = TrajectoryState::new(question, source_tree);
        for operation in &session_log.0 {
            session_state.update(operation.clone());
        }
        session_state
    }
    pub fn collect_session_log_from_tree(tree: &Tree) -> TrajectoryActionLog {
        if tree.current_node_id.is_none() {
            assert!(
                tree.root_node_id.is_none() && tree.nodes.is_empty(),
                "Tree without current node must be empty before root creation"
            );
            return TrajectoryActionLog(Vec::new());
        }
        let mut node_ids_from_current_to_root: Vec<usize> = Vec::new();
        let mut cursor = tree.current_node_id;
        while let Some(node_id) = cursor {
            node_ids_from_current_to_root.push(node_id);
            let node = tree
                .find_node_by_id(node_id)
                .expect("Current path node_id must exist in tree");
            let parent = node.parent_id;
            let root_node_id = tree
                .root_node_id
                .expect("root_node_id must be set when traversing current path");
            assert!(
                parent.is_some() || node.node_id == root_node_id,
                "Non-root node must have a live parent"
            );
            cursor = parent;
        }
        assert!(
            !node_ids_from_current_to_root.is_empty(),
            "Tree traversal should always include at least current node"
        );
        node_ids_from_current_to_root.reverse();

        let mut actions: Vec<RolloutAction> = Vec::new();
        for node_id in node_ids_from_current_to_root {
            let node = tree
                .find_node_by_id(node_id)
                .expect("Path node_id must exist while collecting action log");
            actions.extend(node.step.action_log.iter().cloned());
        }
        TrajectoryActionLog(actions)
    }
    pub fn from_tree(tree: &'a Tree) -> Self {
        let rebuilt_session_log = Self::collect_session_log_from_tree(tree);
        let question = tree.question.clone();
        Self::from_session_log(question, rebuilt_session_log, tree)
    }
    pub fn can_change_plan(&self) -> bool {
        self.num_plan_changes < MAX_PLAN_CHANGES
    }
    pub fn step_quality_accuracy(&self) -> Option<StepQualityAccuracy> {
        assert_eq!(
            self.step_quality_tool_total_count,
            self.step_quality_complete_total_count
        );
        assert_eq!(
            self.step_quality_tool_total_count,
            self.step_quality_focused_total_count
        );
        if self.step_quality_tool_total_count == 0 {
            return None;
        }
        Some(StepQualityAccuracy {
            tool_accuracy: self.step_quality_tool_true_count as f32
                / self.step_quality_tool_total_count as f32,
            complete_accuracy: self.step_quality_complete_true_count as f32
                / self.step_quality_complete_total_count as f32,
            focused_accuracy: self.step_quality_focused_true_count as f32
                / self.step_quality_focused_total_count as f32,
        })
    }
    pub fn can_overwrite_step(&self) -> bool {
        self.step_overwrite_streak < MAX_STEP_OVERWRITE_STREAK
            && self.total_step_overwrites < MAX_TOTAL_STEP_OVERWRITES
    }
    pub fn can_take_action_in_current_step(&self) -> bool {
        self.current_step_num_actions < MAX_ACTIONS_PER_STEP
    }
    pub fn num_additional_actions_allowed_in_current_step(&self) -> usize {
        MAX_ACTIONS_PER_STEP - self.current_step_num_actions
    }
    fn refresh_should_end_session(&mut self) {
        if self.final_answer.is_some() {
            self.should_end_session = true;
            return;
        }
        if self.prev_steps.len() > 20 || self.total_actions > 150 {
            self.final_answer = Some(FORCED_END_MESSAGE.to_string());
            self.should_end_session = true;
            return;
        }
        self.should_end_session = false;
    }

    pub fn update(&mut self, operation: RolloutAction) {
        self.total_actions += 1;
        match operation {
            RolloutAction::PlannerMakeOrChangePlan(plan) => {
                assert!(
                    matches!(
                        self.status,
                        TrajectoryStatus::PlannerMakingOrChangingPlan
                            | TrajectoryStatus::PlannerKeepingCurrentPlan
                    ),
                    "PlannerMakeOrChangePlan can only be called during PlannerMakingOrChangingPlan or PlannerKeepingCurrentPlan"
                );
                match plan {
                    None => {
                        assert_eq!(
                            self.status,
                            TrajectoryStatus::PlannerKeepingCurrentPlan,
                            "PlannerMakeOrChangePlan(None) must be emitted in PlannerKeepingCurrentPlan"
                        );
                    }
                    Some(MakeOrChangePlan::MakePlan(plan_content)) => {
                        assert_eq!(
                            self.status,
                            TrajectoryStatus::PlannerMakingOrChangingPlan,
                            "PlannerMakeOrChangePlan(Some(MakePlan)) must be emitted in PlannerMakingOrChangingPlan"
                        );
                        self.current_plan = Some(plan_content);
                    }
                    Some(MakeOrChangePlan::ChangePlan {
                        plan: new_plan,
                        prev_failed_reason,
                    }) => {
                        assert_eq!(
                            self.status,
                            TrajectoryStatus::PlannerMakingOrChangingPlan,
                            "PlannerMakeOrChangePlan(Some(ChangePlan)) must be emitted in PlannerMakingOrChangingPlan"
                        );
                        assert!(
                            self.can_change_plan(),
                            "Exceed maximum number of plan changes"
                        );
                        let old_plan = self
                            .current_plan
                            .clone()
                            .take()
                            .expect("There must be an existing plan before changing plan");
                        let failed_attempt = FailedAttempt {
                            plan: old_plan,
                            reason: prev_failed_reason,
                        };
                        self.failed_attempts.push(failed_attempt);
                        self.prev_steps.clear();
                        self.num_plan_changes += 1;
                        self.current_plan = Some(new_plan);
                    }
                }
                assert!(
                    self.current_plan.is_some(),
                    "A plan must exist after PlannerMakeOrChangePlan"
                );
                self.status = TrajectoryStatus::PlannerWorkingOnStep;
            }
            RolloutAction::PlannerDecideNextStep(mode) => {
                self.planner_chosen_mode = Some(mode.clone());
                match mode {
                    NextStepDecision::ChangePlan(_reason) => {
                        self.step_overwrite_streak = 0;
                        self.status = TrajectoryStatus::PlannerMakingOrChangingPlan;
                    }
                    NextStepDecision::OverwriteLastStep(_overwrite_reason) => {
                        assert!(
                            self.can_overwrite_step(),
                            "Exceed maximum step overwrite limit"
                        );
                        self.step_overwrite_streak += 1;
                        self.total_step_overwrites += 1;
                        if self.current_plan.is_none() {
                            self.status = TrajectoryStatus::PlannerMakingOrChangingPlan;
                        } else {
                            self.status = TrajectoryStatus::PlannerKeepingCurrentPlan;
                        }
                    }
                    NextStepDecision::Continue => {
                        self.step_overwrite_streak = 0;
                        if self.current_plan.is_none() {
                            self.status = TrajectoryStatus::PlannerMakingOrChangingPlan;
                        } else {
                            self.status = TrajectoryStatus::PlannerKeepingCurrentPlan;
                        }
                    }
                }
            }
            RolloutAction::PlannerReasoning(reasoning) => {
                assert!(
                    self.can_take_action_in_current_step(),
                    "Exceed maximum number of actions in the current step"
                );
                self.current_step_num_actions += 1;
                self.current_step_content_raw.push_str(&reasoning);
                if let Some(boxed_answer) = extract_boxed_content(&reasoning) {
                    self.final_answer = Some(boxed_answer);
                }
            }
            RolloutAction::PlannerToolCall(tool_call) => {
                assert!(
                    self.can_take_action_in_current_step(),
                    "Exceed maximum number of actions in the current step"
                );
                self.current_step_num_actions += 1;
                self.current_step_content_raw.push_str(&tool_call);
            }
            RolloutAction::ToolCallResponse(tool_response) => {
                assert!(
                    self.can_take_action_in_current_step(),
                    "Exceed maximum number of actions in the current step"
                );
                self.current_step_num_actions += 1;
                assert_eq!(
                    self.status,
                    TrajectoryStatus::PlannerWorkingOnStep,
                    "ToolCallResponse can only be called during PlannerTurn"
                );
                match &tool_response {
                    ToolResponse::PythonSuccess(_output) => {
                        self.current_step_last_python_error = None;
                    }
                    ToolResponse::PythonError(error) => {
                        self.current_step_last_python_error = Some(error.clone());
                    }
                    ToolResponse::Intervention(_content) => {
                        self.current_step_last_python_error = None;
                    }
                }
                if let ToolResponse::Intervention(content) = &tool_response {
                    if content == CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE
                        || content == IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE
                        || content == REPETITION_ABORT_MESSAGE
                    {
                        self.should_end_session = true;
                    }
                }
                self.current_step_content_raw
                    .push_str(&tool_response.to_raw_content());
            }
            RolloutAction::PlannerEndStep => {
                assert_eq!(
                    self.status,
                    TrajectoryStatus::PlannerWorkingOnStep,
                    "PlannerEndStep can only be called during PlannerTurn"
                );
                self.current_step_num_actions = 0;
                self.current_step_last_python_error = None;
                self.status = TrajectoryStatus::PlannerCompactingStep;
            }
            RolloutAction::PlannerCompactStep {
                summary,
                step_quality,
            } => {
                assert_eq!(
                    self.status,
                    TrajectoryStatus::PlannerCompactingStep,
                    "PlannerCompactStep can only be called after PlannerEndStep"
                );
                // this will conclude the current step
                self.total_actual_steps += 1;
                if let Some(boxed_answer) = extract_boxed_content(&summary) {
                    self.final_answer = Some(boxed_answer);
                }
                if self.final_answer.is_some() {
                    self.current_step_last_python_error = None;
                }
                if let Some(step_quality) = &step_quality {
                    self.step_quality_tool_total_count += 1;
                    self.step_quality_complete_total_count += 1;
                    self.step_quality_focused_total_count += 1;
                    if step_quality.tool {
                        self.step_quality_tool_true_count += 1;
                    }
                    if step_quality.complete {
                        self.step_quality_complete_true_count += 1;
                    }
                    if step_quality.focused {
                        self.step_quality_focused_true_count += 1;
                    }
                }
                self.current_step_content_compacted = Some(summary);
                self.current_step_quality = step_quality;
                self.status = TrajectoryStatus::PlannerUpdatingPlan;
            }
            RolloutAction::PlannerUpdatePlan(updated_plan) => {
                assert_eq!(
                    self.status,
                    TrajectoryStatus::PlannerUpdatingPlan,
                    "PlannerUpdatePlan can only be called after PlannerCompactStep"
                );
                self.current_plan = Some(updated_plan);
                let step_mode = self
                    .planner_chosen_mode
                    .take()
                    .expect("Planner must have chosen a mode before planner updates plan");
                match &step_mode {
                    NextStepDecision::Continue => {
                        let new_step = CompletedStep::new(
                            step_mode.clone(),
                            self.current_step_content_raw.clone(),
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            self.current_step_quality.take(),
                            None,
                        );
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                        self.current_step_quality = None;
                    }
                    NextStepDecision::OverwriteLastStep(_overwrite_reason) => {
                        let new_step = CompletedStep::new(
                            step_mode.clone(),
                            self.current_step_content_raw.clone(),
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            self.current_step_quality.take(),
                            None,
                        );
                        self.prev_steps.pop();
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                        self.current_step_quality = None;
                    }
                    NextStepDecision::ChangePlan(_) => {
                        let new_step = CompletedStep::new(
                            NextStepDecision::Continue,
                            self.current_step_content_raw.clone(),
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            self.current_step_quality.take(),
                            None,
                        );
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                        self.current_step_quality = None;
                    }
                }
                self.status = TrajectoryStatus::VerifierCommenting;
            }
            RolloutAction::VerifierComment(comment) => {
                assert_eq!(
                    self.status,
                    TrajectoryStatus::VerifierCommenting,
                    "VerifierComment can only be called during VerifierCommenting"
                );
                if let Some(last_step) = self.prev_steps.last_mut() {
                    last_step.current_step_verifier_comment = comment;
                    self.status = TrajectoryStatus::PlannerChoosingMode;
                } else {
                    assert!(
                        comment.is_none(),
                        "Verifier comment must be None when there is no previous step"
                    );
                    self.planner_chosen_mode = Some(NextStepDecision::Continue);
                    self.status = TrajectoryStatus::PlannerMakingOrChangingPlan;
                }
            }
        }
        self.refresh_should_end_session();
    }
    pub fn to_history_prev_steps(&self) -> String {
        let (planner_turn, making_plan) = match self.status {
            TrajectoryStatus::PlannerMakingOrChangingPlan => (true, true),
            TrajectoryStatus::PlannerKeepingCurrentPlan => (true, false),
            TrajectoryStatus::PlannerChoosingMode => (true, false), // should see last step verifier comment if there is any
            TrajectoryStatus::PlannerWorkingOnStep => (true, false), // should see last step verifier comment if there is any
            TrajectoryStatus::PlannerCompactingStep => (true, false), // same as working on step
            TrajectoryStatus::PlannerUpdatingPlan => (true, false), // for current step, should see the compacted content only
            TrajectoryStatus::VerifierCommenting => (false, false), // for current step, should see both the raw content and compacted content
        };
        let mut history = String::new();
        for (i, failed_attempt) in self.failed_attempts.iter().enumerate() {
            history.push_str(&format!("Failed Attempt {}:\n", i + 1));
            history.push_str(&format!("Plan: {}\n", failed_attempt.plan));
            history.push_str(&format!("Reason: {}\n", failed_attempt.reason));
        }
        if !making_plan {
            let current_plan = self
                .current_plan
                .as_ref()
                .expect("There should be a current plan when not making plan");
            history.push_str(&format!("Current plan:\n{}\n", current_plan));
        }
        assert!(
            !(making_plan && self.current_plan.is_none()) || self.prev_steps.is_empty(),
            "When making the initial plan, there should be no previous steps"
        );
        for (i, step) in self.prev_steps.iter().enumerate() {
            history.push_str(&format!("Step {}:\n", i + 1));

            history.push_str(&format!("{}\n", step.content_compacted));

            history.push_str(&format!("Step {} ends.\n", i + 1));
            // only show the last verifier comment if it exists
            if planner_turn && i == self.prev_steps.len() - 1 {
                if let Some(comment) = &step.current_step_verifier_comment {
                    history.push_str(&format!(
                        "Verifier comment on step {}: {}\n",
                        i + 1,
                        comment.comment
                    ));
                }
            }
        }
        history
    }
    pub fn total_display_rounds(&self) -> usize {
        self.prev_steps.len()
    }
}
