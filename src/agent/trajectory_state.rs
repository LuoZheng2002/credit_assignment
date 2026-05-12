use core::panic;

use crate::agent::tree::Tree;

use super::trajectory_action::{TrajectoryAction, TrajectoryActionLog};
use super::trajectory_action_types::{
    CompletedStep, FailedAttempt, MakeOrChangePlan, NextStepDecision, ToolResponse,
};
use super::trajectory_status::TrajectoryStatus;
use crate::agent::trajectory_action_types::FinalAnswer;
use crate::constants::{
    MAX_ACTIONS_PER_STEP, MAX_PLAN_CHANGES, MAX_STEP_OVERWRITE_STREAK, MAX_TOTAL_STEP_OVERWRITES,
};

#[derive(Debug, Clone)]
pub struct TrajectoryState<'a> {
    pub source_tree: &'a Tree,
    pub question: String,
    pub prev_steps: Vec<CompletedStep>,
    pub current_plan: Option<String>,
    pub status: TrajectoryStatus,
    pub failed_attempts: Vec<FailedAttempt>,
    // limit stats
    pub num_plan_changes: usize,
    pub step_overwrite_streak: usize,
    pub total_step_overwrites: usize,
    pub current_step_num_actions: usize,
    pub current_step_last_python_error: Option<String>,
    pub total_actions: usize,
    pub total_actual_steps: usize,
    // special state across many statuses
    pub final_answer: Option<FinalAnswer>,
}

impl<'a> TrajectoryState<'a> {
    fn new(question: String, source_tree: &'a Tree) -> Self {
        Self {
            source_tree,
            question,
            prev_steps: Vec::new(),
            // current_step_content_raw: String::new(),
            num_plan_changes: 0,
            current_step_num_actions: 0,
            current_step_last_python_error: None,
            step_overwrite_streak: 0,
            total_step_overwrites: 0,
            // current_step_content_compacted: None,
            current_plan: None,
            status: TrajectoryStatus::VerifierCommenting,
            // planner_chosen_mode: None,
            final_answer: None,
            failed_attempts: Vec::new(),
            total_actions: 0,
            total_actual_steps: 0,
            // should_end_session: false,
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

        let mut actions: Vec<TrajectoryAction> = Vec::new();
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

    // tree and trajectory?
    // currently we're thinking about how to handle the start and stop of a step
    // whether tree state is needed for knowing what to do next
    // at the beginning of the rollout, the tree has no nodes
    // we create a node when a step ends or a trajectory ends and there is a branching node
    // we want to unify the node creation logic
    // we create a node when the trajectory state is either "step finished" or "trajectory finished"
    // they already have different logic
    // we also need the transition from step ended to verifier commenting, currently it is "start new step"

    pub fn update(&mut self, operation: TrajectoryAction) {
        self.total_actions += 1;
        match &operation {
            TrajectoryAction::StartNewStep => {
                assert!(
                    matches!(self.status, TrajectoryStatus::StepEnded),
                    "StartNewStep can only be called after StepEnded status"
                );
                self.status = TrajectoryStatus::VerifierCommenting;
            }
            TrajectoryAction::StartDeterminingBranchingNode => {
                assert!(
                    matches!(self.status, TrajectoryStatus::TrajectoryEnded { .. }),
                    "StartDeterminingBranchingNode can only be called after TrajectoryEnded status"
                );
                self.status = TrajectoryStatus::DeterminingBranchingNode;
            }
            TrajectoryAction::VerifierComment(verifier_comment) => {
                assert_eq!(
                    self.status,
                    TrajectoryStatus::VerifierCommenting,
                    "VerifierComment can only be called during VerifierCommenting"
                );
                self.status = TrajectoryStatus::PlannerChoosingMode {
                    verifier_comment: verifier_comment.clone(),
                };
            }
            TrajectoryAction::PlannerDecideNextStep(mode) => {
                let TrajectoryStatus::PlannerChoosingMode { verifier_comment } =
                    self.status.clone()
                else {
                    panic!(
                        "PlannerDecideNextStep can only be called during PlannerChoosingMode status"
                    );
                };
                // self.planner_chosen_mode = Some(mode.clone());
                match &mode {
                    NextStepDecision::ChangePlan(_reason) => {
                        self.step_overwrite_streak = 0;
                    }
                    NextStepDecision::OverwriteLastStep(_overwrite_reason) => {
                        assert!(
                            self.can_overwrite_step(),
                            "Exceed maximum step overwrite limit"
                        );
                        self.step_overwrite_streak += 1;
                        self.total_step_overwrites += 1;
                        assert!(
                            self.current_plan.is_some(),
                            "There must be an existing plan when overwriting step"
                        );
                    }
                    NextStepDecision::Continue => {
                        self.step_overwrite_streak = 0;
                    }
                };
                self.status = TrajectoryStatus::PlannerMakingOrChangingPlan {
                    planner_chosen_mode: mode.clone(),
                    verifier_comment,
                };
            }
            TrajectoryAction::PlannerMakeOrChangePlan(plan) => {
                let TrajectoryStatus::PlannerMakingOrChangingPlan {
                    planner_chosen_mode,
                    verifier_comment,
                } = self.status.clone()
                else {
                    panic!(
                        "PlannerMakeOrChangePlan can only be called during PlannerMakingOrChangingPlan status"
                    );
                };
                match plan {
                    None => {
                        assert!(
                            self.current_plan.is_some(),
                            "There must be an existing plan when making or changing plan with None"
                        );
                        assert!(matches!(
                            planner_chosen_mode,
                            NextStepDecision::Continue | NextStepDecision::OverwriteLastStep(_)
                        ));
                    }
                    Some(MakeOrChangePlan::MakePlan(plan_content)) => {
                        assert!(
                            self.current_plan.is_none(),
                            "There should be no existing plan when making the initial plan"
                        );
                        assert!(matches!(planner_chosen_mode, NextStepDecision::Continue));
                        self.current_plan = Some(plan_content.clone());
                    }
                    Some(MakeOrChangePlan::ChangePlan {
                        plan: new_plan,
                        prev_failed_reason,
                    }) => {
                        assert!(
                            self.current_plan.is_some(),
                            "There must be an existing plan when making or changing plan with Some(ChangePlan)"
                        );
                        assert!(matches!(
                            planner_chosen_mode,
                            NextStepDecision::ChangePlan(_)
                        ));
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
                            reason: prev_failed_reason.clone(),
                        };
                        self.failed_attempts.push(failed_attempt);
                        self.prev_steps.clear();
                        self.num_plan_changes += 1;
                        self.current_plan = Some(new_plan.clone());
                    }
                }
                assert!(
                    self.current_plan.is_some(),
                    "A plan must exist after PlannerMakeOrChangePlan"
                );
                // state transition
                self.status = TrajectoryStatus::PlannerWorkingOnStep {
                    planner_chosen_mode,
                    verifier_comment,
                    step_content_raw: String::new(),
                };
            }

            TrajectoryAction::PlannerReasoning { reasoning } => {
                assert!(
                    self.can_take_action_in_current_step(),
                    "Exceed maximum number of actions in the current step"
                );
                let TrajectoryStatus::PlannerWorkingOnStep {
                    planner_chosen_mode: _,
                    verifier_comment: _,
                    step_content_raw,
                } = &mut self.status
                else {
                    panic!(
                        "PlannerReasoning can only be called during PlannerWorkingOnStep status"
                    );
                };
                self.current_step_num_actions += 1;
                step_content_raw.push_str(&reasoning);
            }
            TrajectoryAction::PlannerToolCall(tool_call) => {
                assert!(
                    self.can_take_action_in_current_step(),
                    "Exceed maximum number of actions in the current step"
                );
                let TrajectoryStatus::PlannerWorkingOnStep {
                    planner_chosen_mode: _,
                    verifier_comment: _,
                    step_content_raw,
                } = &mut self.status
                else {
                    panic!(
                        "PlannerReasoning can only be called during PlannerWorkingOnStep status"
                    );
                };
                self.current_step_num_actions += 1;
                step_content_raw.push_str(&tool_call);
            }
            TrajectoryAction::ToolCallResponse(tool_response) => {
                assert!(
                    self.can_take_action_in_current_step(),
                    "Exceed maximum number of actions in the current step"
                );
                let TrajectoryStatus::PlannerWorkingOnStep {
                    planner_chosen_mode: _,
                    verifier_comment: _,
                    step_content_raw,
                } = &mut self.status
                else {
                    panic!(
                        "PlannerReasoning can only be called during PlannerWorkingOnStep status"
                    );
                };
                self.current_step_num_actions += 1;
                match &tool_response {
                    ToolResponse::PythonSuccess(_output) => {
                        self.current_step_last_python_error = None;
                    }
                    ToolResponse::PythonError(error) => {
                        self.current_step_last_python_error = Some(error.clone());
                    } // ToolResponse::EmptyMessageHint => {}
                }
                step_content_raw.push_str(&tool_response.to_raw_content());
            }
            TrajectoryAction::SystemInterrupt(content) => {
                let TrajectoryStatus::PlannerWorkingOnStep {
                    planner_chosen_mode,
                    verifier_comment: _,
                    mut step_content_raw,
                } = self.status.clone()
                else {
                    panic!(
                        "PlannerReasoning can only be called during PlannerWorkingOnStep status"
                    );
                };
                step_content_raw.push_str(&content);
                self.status = TrajectoryStatus::CompactorCompactingStep {
                    planner_chosen_mode,
                    step_content_raw,
                };
            }
            TrajectoryAction::PlannerEndStep => {
                let TrajectoryStatus::PlannerWorkingOnStep {
                    planner_chosen_mode,
                    verifier_comment: _,
                    step_content_raw,
                } = self.status.clone()
                else {
                    panic!("PlannerEndStep can only be called during PlannerWorkingOnStep status");
                };
                self.current_step_num_actions = 0;
                self.current_step_last_python_error = None;
                self.status = TrajectoryStatus::CompactorCompactingStep {
                    planner_chosen_mode,
                    step_content_raw,
                };
            }
            TrajectoryAction::CompactorCompactStep {
                step_content_compacted,
                step_quality: _,
            } => {
                let TrajectoryStatus::CompactorCompactingStep {
                    planner_chosen_mode,
                    step_content_raw,
                } = self.status.clone()
                else {
                    panic!(
                        "PlannerCompactStep can only be called during PlannerCompactingStep status"
                    );
                };
                self.status = TrajectoryStatus::PlannerUpdatingPlan {
                    planner_chosen_mode,
                    step_content_raw,
                    step_content_compacted: step_content_compacted.clone(),
                };
            }
            TrajectoryAction::PlannerUpdatePlan(updated_plan) => {
                let TrajectoryStatus::PlannerUpdatingPlan {
                    planner_chosen_mode,
                    step_content_raw,
                    step_content_compacted,
                } = self.status.clone()
                else {
                    panic!(
                        "PlannerUpdatePlan can only be called during PlannerUpdatingPlan status"
                    );
                };
                self.total_actual_steps += 1;
                if updated_plan.is_some() {
                    assert!(
                        self.final_answer.is_none(),
                        "If final answer is found, there should be no updated plan"
                    );
                    self.current_plan = updated_plan.clone();
                }
                let overwrite =
                    matches!(planner_chosen_mode, NextStepDecision::OverwriteLastStep(_));
                let new_step = CompletedStep::new(
                    planner_chosen_mode,
                    step_content_raw,
                    step_content_compacted,
                );
                if overwrite {
                    self.prev_steps.pop();
                }
                self.prev_steps.push(new_step);
                if let Some(final_answer) = self.final_answer.clone() {
                    self.status = TrajectoryStatus::TrajectoryEnded { final_answer }
                } else {
                    self.status = TrajectoryStatus::StepEnded;
                }
            }
            TrajectoryAction::SubmitFinalAnswer(answer) => {
                assert!(
                    self.final_answer.is_none(),
                    "Final answer has already been submitted, cannot submit again"
                );
                self.final_answer = Some(answer.clone());
            }
        }
    }

    pub fn to_history_prev_steps(&self) -> String {
        let making_plan = matches!(
            self.status,
            TrajectoryStatus::PlannerMakingOrChangingPlan { .. }
        );
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
                .cloned()
                .unwrap_or_else(|| "No current plan available.".to_string());
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
        }
        if let Some(comment) = self.status.try_get_verifier_comment() {
            history.push_str(&format!(
                "Verifier comment on step {}: {}\n",
                self.prev_steps.len(),
                comment.comment
            ));
        }
        history
    }

    pub fn total_display_rounds(&self) -> usize {
        self.prev_steps.len()
    }
}
