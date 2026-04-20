use crate::deepmath::parse_answers::extract_boxed_content;

use super::actions::{RolloutAction, ToolResponse, TrajectoryActionLog};
use super::constants::{
    CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE, FORCED_END_MESSAGE,
    IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE, MAX_ACTIONS_PER_STEP, MAX_PLAN_CHANGES,
    MAX_STEP_OVERWRITE_STREAK, MAX_TOTAL_STEP_OVERWRITES, REPETITION_ABORT_MESSAGE,
};
use super::tree::Tree;
use super::types::{
    CompletedStep, FailedAttempt, MakeOrChangePlan, NextStepDecision, TrajectoryStatus,
};

#[derive(Debug, Clone)]
pub struct TrajectoryState<'a> {
    pub source_tree: &'a Tree,
    pub question: String,
    pub prev_steps: Vec<CompletedStep>,
    pub current_step_content_raw: String,
    pub current_step_content_compacted: Option<String>,
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
    pub should_end_session: bool,
}

impl<'a> TrajectoryState<'a> {
    fn new(question: String, source_tree: &'a Tree) -> Self {
        Self {
            source_tree,
            question,
            prev_steps: Vec::new(),
            current_step_content_raw: String::new(),
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
                .nodes
                .get(node_id)
                .expect("Current path node_id must exist in tree");
            assert_eq!(node.node_id, node_id, "Node index must equal node_id");
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
                .nodes
                .get(node_id)
                .expect("Path node_id must exist while collecting action log");
            assert_eq!(node.node_id, node_id, "Node index must equal node_id");
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
                step_quality: _,
            } => {
                assert_eq!(
                    self.status,
                    TrajectoryStatus::PlannerCompactingStep,
                    "PlannerCompactStep can only be called after PlannerEndStep"
                );
                self.total_actual_steps += 1;
                if let Some(boxed_answer) = extract_boxed_content(&summary) {
                    self.final_answer = Some(boxed_answer);
                }
                if self.final_answer.is_some() {
                    self.current_step_last_python_error = None;
                }
                self.current_step_content_compacted = Some(summary);
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
                            None,
                            None,
                        );
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                    }
                    NextStepDecision::OverwriteLastStep(_overwrite_reason) => {
                        let new_step = CompletedStep::new(
                            step_mode.clone(),
                            self.current_step_content_raw.clone(),
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            None,
                            None,
                        );
                        self.prev_steps.pop();
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                    }
                    NextStepDecision::ChangePlan(_) => {
                        let new_step = CompletedStep::new(
                            NextStepDecision::Continue,
                            self.current_step_content_raw.clone(),
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            None,
                            None,
                        );
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
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
            TrajectoryStatus::PlannerChoosingMode => (true, false),
            TrajectoryStatus::PlannerWorkingOnStep => (true, false),
            TrajectoryStatus::PlannerCompactingStep => (true, false),
            TrajectoryStatus::PlannerUpdatingPlan => (true, false),
            TrajectoryStatus::VerifierCommenting => (false, false),
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
