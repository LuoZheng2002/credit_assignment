use serde::{Deserialize, Serialize};

use crate::deepmath::parse_answers::extract_boxed_content;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NextStepDecision {
    Continue,
    OverwriteLastStep(String),
    ChangePlan {
        fail_reason: String,
        possible_future_direction: String,
    },
}

impl NextStepDecision {
    pub fn is_overwriting(&self) -> bool {
        match self {
            NextStepDecision::Continue => false,
            NextStepDecision::OverwriteLastStep(_) => true,
            NextStepDecision::ChangePlan { .. } => panic!(
                "ChangePlan should not be a valid step mode when the planner has entered the working on step status"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisplayPlannerStepComplete {
    // pub step_mode: DisplayStepMode,
    // pub content_raw: String,
    pub content_compacted: String,
    pub current_step_verifier_comment: Option<String>,
}
impl DisplayPlannerStepComplete {
    pub fn new(content_compacted: String, current_step_verifier_comment: Option<String>) -> Self {
        Self {
            content_compacted,
            current_step_verifier_comment,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOperation {
    PlannerMakePlan(String), // this should be a mandatory process instead of a choice
    PlannerDecideNextStep(NextStepDecision),
    PlannerReasoning(String),
    PlannerToolCall(String),  // with <tool_call> ... </tool_call> wrapper
    ToolCallResponse(String), // with <tool_response> ... </tool_response> wrapper
    PlannerEndStep,
    PlannerCompactStep(String),
    PlannerUpdatePlan(String),
    VerifierComment(Option<String>),
}

impl ModelOperation {
    pub fn to_pretty_string(&self) -> String {
        match self {
            ModelOperation::PlannerMakePlan(plan) => format!("[PlannerMakePlan]:\n{}", plan),
            ModelOperation::PlannerDecideNextStep(mode) => {
                format!("[PlannerChooseMode]: {:?}", mode)
            }
            ModelOperation::PlannerReasoning(reasoning) => {
                format!("[PlannerReasoning]:\n{}", reasoning)
            }
            ModelOperation::PlannerToolCall(tool_call) => {
                format!("[PlannerToolCall]:\n{}", tool_call)
            }
            ModelOperation::PlannerEndStep => "[PlannerEndStep]".to_string(),
            ModelOperation::PlannerCompactStep(compacted) => {
                format!("[PlannerCompactStep]:\n{}", compacted)
            }
            ModelOperation::PlannerUpdatePlan(updated_plan) => {
                format!("[PlannerUpdatePlan]:\n{}", updated_plan)
            }
            ModelOperation::ToolCallResponse(tool_response) => {
                format!("[ToolCallResponse]:\n{}", tool_response)
            }
            ModelOperation::VerifierComment(comment) => {
                format!("[VerifierComment]:\n{:?}", comment)
            }
        }
    }
    pub fn to_concise_string(&self) -> String {
        match self {
            ModelOperation::PlannerMakePlan(_plan) => format!("[PlannerMakePlan]"),
            ModelOperation::PlannerDecideNextStep(mode) => {
                format!("[PlannerChooseMode]: {:?}", mode)
            }
            ModelOperation::PlannerReasoning(_reasoning) => "[PlannerReasoning]".to_string(),
            ModelOperation::PlannerToolCall(_tool_call) => "[PlannerToolCall]".to_string(),
            ModelOperation::PlannerEndStep => "[PlannerEndStep]".to_string(),
            ModelOperation::PlannerCompactStep(_compacted) => "[PlannerCompactStep]".to_string(),
            ModelOperation::PlannerUpdatePlan(_updated_plan) => "[PlannerUpdatePlan]".to_string(),
            ModelOperation::ToolCallResponse(_tool_response) => "[ToolCallResponse]".to_string(),
            ModelOperation::VerifierComment(comment) => format!(
                "[VerifierComment]\n{}",
                comment.clone().unwrap_or_else(|| "None".to_string())
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLog {
    pub parsed_operations: Vec<ModelOperation>,
    pub model_raw_outputs: Vec<String>,
}

impl SessionLog {
    pub fn total_actual_rounds(&self) -> usize {
        self.parsed_operations
            .iter()
            .filter(|op| matches!(op, ModelOperation::PlannerDecideNextStep(_)))
            .count()
    }

    pub fn operations(&self) -> &[ModelOperation] {
        &self.parsed_operations
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    PlannerMakingPlan, // this should be a mandatory process instead of a choice
    PlannerChoosingMode,
    PlannerWorkingOnStep,
    PlannerCompactingStep,
    PlannerUpdatingPlan,
    VerifierCommenting,
}

#[derive(Debug, Clone)]
pub struct FailedAttempt {
    pub plan: String,
    pub fail_reason: String,
    pub possible_future_direction: String,
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub question: String,
    pub prev_steps: Vec<DisplayPlannerStepComplete>,
    pub current_step_content_raw: String,
    pub current_step_content_compacted: Option<String>,
    pub current_step_verifier_comment: Option<String>,
    pub current_plan: Option<String>,
    pub session_status: SessionStatus,
    pub planner_chosen_mode: Option<NextStepDecision>,
    pub final_answer: Option<String>,
    pub failed_attempts: Vec<FailedAttempt>,
    pub num_plan_changes: usize,
    pub step_overwrite_streak: usize,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub session_log: SessionLog,
    pub session_state: SessionState,
}

impl Session {
    pub fn new(question: String) -> Self {
        Self {
            session_log: SessionLog {
                parsed_operations: Vec::new(),
                model_raw_outputs: Vec::new(),
            },
            session_state: SessionState::new(question),
        }
    }
    pub fn apply_parsed_operation(&mut self, operation: ModelOperation) -> bool {
        self.session_log.parsed_operations.push(operation.clone());
        self.session_state.update(operation)
    }
    pub fn add_model_raw_output(&mut self, raw_output: String) {
        self.session_log.model_raw_outputs.push(raw_output);
    }
}

pub const MAX_PLAN_CHANGES: usize = 2;
pub const MAX_STEP_OVERWRITE_STREAK: usize = 2;
impl SessionState {
    pub fn new(question: String) -> Self {
        Self {
            question,
            prev_steps: Vec::new(),
            current_step_content_raw: String::new(),
            current_step_verifier_comment: None,
            num_plan_changes: 0,
            step_overwrite_streak: 0,
            current_step_content_compacted: None,
            current_plan: None,
            session_status: SessionStatus::PlannerMakingPlan,
            planner_chosen_mode: None,
            final_answer: None,
            failed_attempts: Vec::new(),
        }
    }
    pub fn from_session_log(question: String, session_log: &SessionLog) -> Self {
        let mut session_state = SessionState::new(question);
        for operation in &session_log.parsed_operations {
            session_state.update(operation.clone());
        }
        session_state
    }
    pub fn can_change_plan(&self) -> bool {
        self.num_plan_changes < MAX_PLAN_CHANGES
    }
    pub fn can_overwrite_step(&self) -> bool {
        self.step_overwrite_streak < MAX_STEP_OVERWRITE_STREAK
    }
    pub fn update(&mut self, operation: ModelOperation) -> bool {
        let mut should_end_session = false;
        match operation {
            ModelOperation::PlannerMakePlan(plan) => {
                self.current_plan = Some(plan);
                self.session_status = SessionStatus::PlannerChoosingMode;
            }
            ModelOperation::PlannerDecideNextStep(mode) => {
                self.planner_chosen_mode = Some(mode.clone());
                match mode {
                    NextStepDecision::ChangePlan {
                        fail_reason,
                        possible_future_direction,
                    } => {
                        let old_plan = self
                            .current_plan
                            .clone()
                            .take()
                            .expect("There must be a plan to change");
                        let failed_attempt = FailedAttempt {
                            plan: old_plan,
                            fail_reason: fail_reason.clone(),
                            possible_future_direction: possible_future_direction.clone(),
                        };
                        self.failed_attempts.push(failed_attempt);
                        self.prev_steps.clear();
                        assert!(
                            self.can_change_plan(),
                            "Exceed maximum number of plan changes"
                        );
                        self.num_plan_changes += 1;
                        self.session_status = SessionStatus::PlannerMakingPlan;
                    }
                    NextStepDecision::OverwriteLastStep(_overwrite_reason) => {
                        assert!(
                            self.can_overwrite_step(),
                            "Exceed maximum step overwrite streak"
                        );
                        self.step_overwrite_streak += 1;
                        self.session_status = SessionStatus::PlannerWorkingOnStep;
                    }
                    NextStepDecision::Continue => {
                        self.step_overwrite_streak = 0;
                        self.session_status = SessionStatus::PlannerWorkingOnStep;
                    }
                }
            }
            ModelOperation::PlannerReasoning(reasoning) => {
                self.current_step_content_raw.push_str(&reasoning);
                if let Some(boxed_answer) = extract_boxed_content(&self.current_step_content_raw) {
                    self.final_answer = Some(boxed_answer);
                    should_end_session = true;
                }
            }
            ModelOperation::PlannerToolCall(tool_call) => {
                self.current_step_content_raw.push_str(&tool_call);
            }
            ModelOperation::ToolCallResponse(tool_response) => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerWorkingOnStep,
                    "ToolCallResponse can only be called during PlannerTurn"
                );
                self.current_step_content_raw.push_str(&tool_response);
            }
            ModelOperation::PlannerEndStep => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerWorkingOnStep,
                    "PlannerEndStep can only be called during PlannerTurn"
                );
                self.session_status = SessionStatus::PlannerCompactingStep;
            }
            ModelOperation::PlannerCompactStep(compacted) => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerCompactingStep,
                    "PlannerCompactStep can only be called after PlannerEndStep"
                );
                self.current_step_content_compacted = Some(compacted);
                self.session_status = SessionStatus::PlannerUpdatingPlan;
            }
            ModelOperation::PlannerUpdatePlan(updated_plan) => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerUpdatingPlan,
                    "PlannerUpdatePlan can only be called after PlannerCompactStep"
                );
                self.current_plan = Some(updated_plan);
                self.session_status = SessionStatus::VerifierCommenting;
            }
            ModelOperation::VerifierComment(comment) => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::VerifierCommenting,
                    "VerifierComment can only be called during VerifierCommenting"
                );
                // this operation flushes the current step and moves it to prev_steps
                self.current_step_verifier_comment = comment;
                let step_mode = self
                    .planner_chosen_mode
                    .take()
                    .expect("Planner must have chosen a mode before verifier commenting");
                match step_mode {
                    NextStepDecision::Continue => {
                        let new_step = DisplayPlannerStepComplete::new(
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            self.current_step_verifier_comment.take(),
                        );
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                        self.current_step_verifier_comment = None;
                    }
                    NextStepDecision::OverwriteLastStep(_overwrite_reason) => {
                        let new_step = DisplayPlannerStepComplete::new(
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            self.current_step_verifier_comment.take(),
                        );
                        self.prev_steps.pop();
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                        self.current_step_verifier_comment = None;
                    }
                    NextStepDecision::ChangePlan { .. } => {
                        panic!(
                            "ChangePlan should not be a valid step mode when the planner has entered the working on step status."
                        )
                    }
                }
                self.session_status = SessionStatus::PlannerChoosingMode;
            }
        }
        should_end_session
    }
    pub fn to_history_prev_steps(&self) -> String {
        let (planner_turn, making_plan) = match self.session_status {
            SessionStatus::PlannerMakingPlan => (true, true), // it should only show the attempts, but not steps
            SessionStatus::PlannerChoosingMode => (true, false), // should see last step verifier comment if there is any
            SessionStatus::PlannerWorkingOnStep => (true, false), // should see last step verifier comment if there is any
            SessionStatus::PlannerCompactingStep => (true, false), // same as working on step
            SessionStatus::PlannerUpdatingPlan => (true, false), // for current step, should see the compacted content only
            SessionStatus::VerifierCommenting => (false, false), // for current step, should see both the raw content and compacted content
        };
        let mut history = String::new();
        for (i, failed_attempt) in self.failed_attempts.iter().enumerate() {
            history.push_str(&format!("Failed Attempt {}:\n", i + 1));
            history.push_str(&format!("Plan: {}\n", failed_attempt.plan));
            history.push_str(&format!("Fail reason: {}\n", failed_attempt.fail_reason));
            history.push_str(&format!(
                "Possible future direction: {}\n",
                failed_attempt.possible_future_direction
            ));
        }
        if !making_plan {
            let current_plan = self
                .current_plan
                .as_ref()
                .expect("There should be a current plan when not making plan");
            history.push_str(&format!("Current plan:\n{}\n", current_plan));
        }
        assert!(
            !making_plan || self.prev_steps.is_empty(),
            "When making plan, there should be no previous steps"
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
                        comment
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
