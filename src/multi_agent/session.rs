use serde::{Deserialize, Serialize};

use crate::deepmath::parse_answers::extract_boxed_content;
use crate::multi_agent::generate_rollout_answers::StepQualityAccuracy;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NextStepDecision {
    Continue,
    OverwriteLastStep(String),
    ChangePlan(String),
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
    pub current_step_verifier_comment: Option<String>,
}
impl CompletedStep {
    pub fn new(
        current_step_mode: NextStepDecision,
        content_raw: String,
        content_compacted: String,
        step_quality: Option<StepQuality>,
        current_step_verifier_comment: Option<String>,
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
    PlannerMakePlan(String), // this should be a mandatory process instead of a choice
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
    VerifierComment(Option<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutActionLogItem {
    pub question_id: usize,
    pub action: RolloutAction,
}

impl RolloutAction {
    pub fn to_pretty_string(&self) -> String {
        match self {
            RolloutAction::PlannerMakePlan(plan) => format!("[PlannerMakePlan]:\n{}", plan),
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
            RolloutAction::PlannerMakePlan(_plan) => format!("[PlannerMakePlan]"),
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
            RolloutAction::VerifierComment(comment) => format!(
                "[VerifierComment]\n{}",
                comment.clone().unwrap_or_else(|| "None".to_string())
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLog(pub Vec<RolloutAction>);

impl SessionLog {
    pub fn total_actual_rounds(&self) -> usize {
        self.0
            .iter()
            .filter(|op| matches!(op, RolloutAction::PlannerDecideNextStep(_)))
            .count()
    }

    pub fn step_quality_accuracy(&self) -> Option<StepQualityAccuracy> {
        let mut tool_true_count = 0usize;
        let mut tool_total_count = 0usize;
        let mut complete_true_count = 0usize;
        let mut complete_total_count = 0usize;
        let mut focused_true_count = 0usize;
        let mut focused_total_count = 0usize;
        for action in &self.0 {
            if let RolloutAction::PlannerCompactStep {
                step_quality: Some(step_quality),
                ..
            } = action
            {
                tool_total_count += 1;
                if step_quality.tool {
                    tool_true_count += 1;
                }

                complete_total_count += 1;
                if step_quality.complete {
                    complete_true_count += 1;
                }

                focused_total_count += 1;
                if step_quality.focused {
                    focused_true_count += 1;
                }
            }
        }
        assert_eq!(tool_total_count, complete_total_count);
        assert_eq!(tool_total_count, focused_total_count);
        if tool_total_count == 0 {
            return None;
        }

        Some(StepQualityAccuracy {
            tool_accuracy: tool_true_count as f32 / tool_total_count as f32,
            complete_accuracy: complete_true_count as f32 / complete_total_count as f32,
            focused_accuracy: focused_true_count as f32 / focused_total_count as f32,
        })
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
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub question: String,
    pub prev_steps: Vec<CompletedStep>,
    pub current_step_content_raw: String,
    pub current_step_content_compacted: Option<String>,
    pub current_step_quality: Option<StepQuality>,
    pub current_step_verifier_comment: Option<String>,
    pub current_plan: Option<String>,
    pub session_status: SessionStatus,
    pub planner_chosen_mode: Option<NextStepDecision>,
    pub final_answer: Option<String>,
    pub failed_attempts: Vec<FailedAttempt>,
    pub num_plan_changes: usize,
    pub step_overwrite_streak: usize,
    pub current_step_num_actions: usize,
    pub current_step_last_python_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub session_log: SessionLog,
    pub session_state: SessionState,
}

impl Session {
    pub fn new(question: String) -> Self {
        Self {
            session_log: SessionLog(Vec::new()),
            session_state: SessionState::new(question),
        }
    }
    pub fn apply_parsed_operation(&mut self, operation: RolloutAction) -> bool {
        self.session_log.0.push(operation.clone());
        self.session_state.update(operation)
    }
}

pub const MAX_PLAN_CHANGES: usize = 2;
pub const MAX_STEP_OVERWRITE_STREAK: usize = 2;
pub const MAX_ACTIONS_PER_STEP: usize = 30;
impl SessionState {
    pub fn new(question: String) -> Self {
        Self {
            question,
            prev_steps: Vec::new(),
            current_step_content_raw: String::new(),
            current_step_verifier_comment: None,
            current_step_quality: None,
            num_plan_changes: 0,
            current_step_num_actions: 0,
            current_step_last_python_error: None,
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
        for operation in &session_log.0 {
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
    pub fn can_take_action_in_current_step(&self) -> bool {
        self.current_step_num_actions < MAX_ACTIONS_PER_STEP
    }
    pub fn num_additional_actions_allowed_in_current_step(&self) -> usize {
        MAX_ACTIONS_PER_STEP - self.current_step_num_actions
    }
    pub fn update(&mut self, operation: RolloutAction) -> bool {
        let mut should_end_session = false;
        match operation {
            RolloutAction::PlannerMakePlan(plan) => {
                self.current_plan = Some(plan);
                self.session_status = SessionStatus::PlannerChoosingMode;
            }
            RolloutAction::PlannerDecideNextStep(mode) => {
                self.planner_chosen_mode = Some(mode.clone());
                match mode {
                    NextStepDecision::ChangePlan(reason) => {
                        let old_plan = self
                            .current_plan
                            .clone()
                            .take()
                            .expect("There must be a plan to change");
                        let failed_attempt = FailedAttempt {
                            plan: old_plan,
                            reason: reason.clone(),
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
            RolloutAction::PlannerReasoning(reasoning) => {
                assert!(
                    self.can_take_action_in_current_step(),
                    "Exceed maximum number of actions in the current step"
                );
                self.current_step_num_actions += 1;
                self.current_step_content_raw.push_str(&reasoning);
                if let Some(boxed_answer) = extract_boxed_content(&reasoning) {
                    self.final_answer = Some(boxed_answer);
                    // should_end_session = true;
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
                    self.session_status,
                    SessionStatus::PlannerWorkingOnStep,
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
                self.current_step_content_raw
                    .push_str(&tool_response.to_raw_content());
            }
            RolloutAction::PlannerEndStep => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerWorkingOnStep,
                    "PlannerEndStep can only be called during PlannerTurn"
                );
                self.current_step_num_actions = 0;
                self.current_step_last_python_error = None;
                self.session_status = SessionStatus::PlannerCompactingStep;
            }
            RolloutAction::PlannerCompactStep {
                summary,
                step_quality,
            } => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerCompactingStep,
                    "PlannerCompactStep can only be called after PlannerEndStep"
                );
                if let Some(boxed_answer) = extract_boxed_content(&summary) {
                    self.final_answer = Some(boxed_answer);
                }
                if self.final_answer.is_some() {
                    self.current_step_last_python_error = None;
                    should_end_session = true;
                }
                self.current_step_content_compacted = Some(summary);
                self.current_step_quality = step_quality;
                self.session_status = SessionStatus::PlannerUpdatingPlan;
            }
            RolloutAction::PlannerUpdatePlan(updated_plan) => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerUpdatingPlan,
                    "PlannerUpdatePlan can only be called after PlannerCompactStep"
                );
                self.current_plan = Some(updated_plan);
                self.session_status = SessionStatus::VerifierCommenting;
            }
            RolloutAction::VerifierComment(comment) => {
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
                match &step_mode {
                    NextStepDecision::Continue => {
                        let new_step = CompletedStep::new(
                            step_mode.clone(),
                            self.current_step_content_raw.clone(),
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            self.current_step_quality.take(),
                            self.current_step_verifier_comment.take(),
                        );
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                        self.current_step_quality = None;
                        self.current_step_verifier_comment = None;
                    }
                    NextStepDecision::OverwriteLastStep(_overwrite_reason) => {
                        let new_step = CompletedStep::new(
                            step_mode.clone(),
                            self.current_step_content_raw.clone(),
                            self.current_step_content_compacted
                                .take()
                                .expect("The compacted content must be available"),
                            self.current_step_quality.take(),
                            self.current_step_verifier_comment.take(),
                        );
                        self.prev_steps.pop();
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_content_compacted = None;
                        self.current_step_quality = None;
                        self.current_step_verifier_comment = None;
                    }
                    NextStepDecision::ChangePlan(_) => {
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
