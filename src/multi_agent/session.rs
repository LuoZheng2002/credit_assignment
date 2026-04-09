use serde::{Deserialize, Serialize};

use crate::deepmath::parse_answers::extract_boxed_content;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepDirection {
    Proceed,
    ChangePlan,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActualStepMode {
    Append(StepDirection),
    OverwriteLastStep(StepDirection),
    Compact,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisplayStepMode {
    Directional(StepDirection),
    Compacted,
}

#[derive(Debug, Clone)]
pub struct ToolCallSegment {
    pub tool_call_string: String,
    pub tool_result_string: String,
}

#[derive(Debug, Clone)]
pub enum PlannerStepOperation {
    Reasoning(String),
    ToolCall(String),
}

#[derive(Debug, Clone)]
pub struct DisplayPlannerStepComplete {
    pub step_mode: DisplayStepMode,
    pub content_raw: String,
    pub current_step_verifier_comment: Option<String>,
}
impl DisplayPlannerStepComplete {
    pub fn new(
        current_step_verifier_comment: Option<String>,
        step_mode: DisplayStepMode,
        content_raw: String,
    ) -> Self {
        Self {
            current_step_verifier_comment,
            step_mode,
            content_raw,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOperation {
    PlannerChooseMode(ActualStepMode),
    PlannerReasoning(String),
    PlannerToolCall(String), // with <tool_call> ... </tool_call> wrapper
    PlannerEndStep,
    ToolCallResponse(String), // with <tool_response> ... </tool_response> wrapper
    VerifierComment(Option<String>),
}

impl ModelOperation {
    pub fn to_pretty_string(&self) -> String {
        match self {
            ModelOperation::PlannerChooseMode(mode) => format!("[PlannerChooseMode]: {:?}", mode),
            ModelOperation::PlannerReasoning(reasoning) => {
                format!("[PlannerReasoning]:\n{}", reasoning)
            }
            ModelOperation::PlannerToolCall(tool_call) => {
                format!("[PlannerToolCall]:\n{}", tool_call)
            }
            ModelOperation::PlannerEndStep => "[PlannerEndStep]".to_string(),
            ModelOperation::ToolCallResponse(tool_response) => {
                format!("[ToolCallResponse]:\n{}", tool_response)
            }
            ModelOperation::VerifierComment(comment) => format!("[VerifierComment]: {:?}", comment),
        }
    }
    pub fn to_concise_string(&self) -> String {
        match self {
            ModelOperation::PlannerChooseMode(mode) => format!("[PlannerChooseMode]: {:?}", mode),
            ModelOperation::PlannerReasoning(_reasoning) => "[PlannerReasoning]".to_string(),
            ModelOperation::PlannerToolCall(_tool_call) => "[PlannerToolCall]".to_string(),
            ModelOperation::PlannerEndStep => "[PlannerEndStep]".to_string(),
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
            .filter(|op| matches!(op, ModelOperation::PlannerChooseMode(_)))
            .count()
    }

    pub fn operations(&self) -> &[ModelOperation] {
        &self.parsed_operations
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerStatus {
    PlannerChoosingMode,
    PlannerChosen(ActualStepMode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    PlannerTurn,
    VerifierTurn,
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub prev_steps: Vec<DisplayPlannerStepComplete>,
    pub current_step_content_raw: String,
    pub current_step_verifier_comment: Option<String>,
    pub planner_status: PlannerStatus,
    pub session_status: SessionStatus,
    pub final_answer: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub session_log: SessionLog,
    pub session_state: SessionState,
}

impl Session {
    pub fn new() -> Self {
        Self {
            session_log: SessionLog {
                parsed_operations: Vec::new(),
                model_raw_outputs: Vec::new(),
            },
            session_state: SessionState::new(),
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

impl SessionState {
    pub fn new() -> Self {
        Self {
            prev_steps: Vec::new(),
            current_step_content_raw: String::new(),
            current_step_verifier_comment: None,
            session_status: SessionStatus::PlannerTurn,
            planner_status: PlannerStatus::PlannerChoosingMode,
            final_answer: None,
        }
    }
    pub fn from_session_log(session_log: &SessionLog) -> Self {
        let mut session_state = SessionState::new();
        for operation in &session_log.parsed_operations {
            session_state.update(operation.clone());
        }
        session_state
    }
    pub fn update(&mut self, operation: ModelOperation) -> bool {
        let mut should_end_session = false;
        match operation {
            ModelOperation::PlannerChooseMode(mode) => {
                self.planner_status = PlannerStatus::PlannerChosen(mode);
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
            ModelOperation::PlannerEndStep => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::PlannerTurn,
                    "PlannerEndStep can only be called during PlannerTurn"
                );
                assert!(
                    matches!(self.planner_status, PlannerStatus::PlannerChosen(_)),
                    "Invalid state transition: PlannerEndStep can only be called after PlannerChosen"
                );
                self.session_status = SessionStatus::VerifierTurn;
            }
            ModelOperation::ToolCallResponse(tool_response) => {
                self.current_step_content_raw.push_str(&tool_response);
            }
            ModelOperation::VerifierComment(comment) => {
                assert_eq!(
                    self.session_status,
                    SessionStatus::VerifierTurn,
                    "VerifierComment can only be called during VerifierTurn"
                );
                let PlannerStatus::PlannerChosen(step_mode) = self.planner_status else {
                    panic!("Invalid state: PlannerChosen must be set before VerifierComment");
                };
                self.current_step_verifier_comment = comment;
                match step_mode {
                    ActualStepMode::Append(direction) => {
                        let display_step_mode = DisplayStepMode::Directional(direction);
                        let new_step = DisplayPlannerStepComplete::new(
                            self.current_step_verifier_comment.take(),
                            display_step_mode,
                            self.current_step_content_raw.clone(),
                        );
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_verifier_comment = None;
                    }
                    ActualStepMode::OverwriteLastStep(direction) => {
                        let display_step_mode = DisplayStepMode::Directional(direction);
                        let new_step = DisplayPlannerStepComplete::new(
                            self.current_step_verifier_comment.take(),
                            display_step_mode,
                            self.current_step_content_raw.clone(),
                        );
                        self.prev_steps.pop();
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_verifier_comment = None;
                    }
                    ActualStepMode::Compact => {
                        let display_step_mode = DisplayStepMode::Compacted;
                        // we do not keep verifier comment in compact mode
                        let new_step = DisplayPlannerStepComplete::new(
                            self.current_step_verifier_comment.take(),
                            display_step_mode,
                            self.current_step_content_raw.clone(),
                        );
                        // in compact mode, we clear all previous steps and only keep the current step
                        self.prev_steps.clear();
                        self.prev_steps.push(new_step);
                        self.current_step_content_raw.clear();
                        self.current_step_verifier_comment = None;
                    }
                }
                self.planner_status = PlannerStatus::PlannerChoosingMode;
                self.session_status = SessionStatus::PlannerTurn;
            }
        }
        should_end_session
    }
    pub fn to_history_prev_steps(&self, planner_turn: bool) -> String {
        let mut history = String::new();
        for (i, step) in self.prev_steps.iter().enumerate() {
            history.push_str(&format!("Step {}:\n", i + 1));
            let step_mode_str = match step.step_mode {
                DisplayStepMode::Directional(direction) => match direction {
                    StepDirection::Proceed => {
                        "This step is a continuation of the previous reasoning direction."
                    }
                    StepDirection::ChangePlan => {
                        "This step is attempting a different reasoning direction from the previous steps."
                    }
                },
                DisplayStepMode::Compacted => "This step is a compacted summary of previous steps.",
            };

            history.push_str(&format!("Current step mode: {}\n", step_mode_str));

            history.push_str(&format!("Assistant: {}\n", step.content_raw));

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

        match self.session_status {
            SessionStatus::PlannerTurn => {
                assert_eq!(
                    planner_turn, true,
                    "to_history should be called with planner_turn=true when it's planner's turn"
                );
            }
            SessionStatus::VerifierTurn => {
                assert_eq!(
                    planner_turn, false,
                    "to_history should be called with planner_turn=false when it's verifier's turn"
                );
            }
        }
        history
    }
    // pub fn to_history_curr_step(&self, planner_turn: bool) -> String {
    //     let mut history = String::new();
    //     let current_step_index = self.prev_steps.len() + 1;
    //     let current_step_hint_str = if planner_turn {
    //         format!("Current step {} (not yet completed):\n", current_step_index)
    //     } else {
    //         format!(
    //             "Current step {} (that you're about to evaluate):\n",
    //             current_step_index
    //         )
    //     };
    //     history.push_str(&current_step_hint_str);

    //     let planner_status_description: &str = match self.planner_status {
    //         PlannerStatus::PlannerChoosingMode => {
    //             "Assistant is currently deciding how to proceed with the current step."
    //         }
    //         PlannerStatus::PlannerChosen(step_mode) => match step_mode {
    //             ActualStepMode::Append(direction) => match direction {
    //                 StepDirection::Proceed => {
    //                     "Assistant has chosen to continue the previous reasoning direction."
    //                 }
    //                 StepDirection::ChangePlan => {
    //                     "Assistant has chosen to attempt a different reasoning direction from the previous steps."
    //                 }
    //             },
    //             ActualStepMode::OverwriteLastStep(direction) => match direction {
    //                 StepDirection::Proceed => {
    //                     "Assistant has chosen to OVERWRITE the last step while maintaining its reasoning direction."
    //                 }
    //                 StepDirection::ChangePlan => {
    //                     "Assistant has chosen to OVERWRITE the last step while changing its reasoning direction."
    //                 }
    //             },
    //             ActualStepMode::Compact => {
    //                 "Assistant has chosen to COMPACT all previous steps into a more concise form."
    //             }
    //             ActualStepMode::SubmitAnswer => "Assistant has chosen to SUBMIT the final answer.",
    //         },
    //     };
    //     history.push_str(&format!("{}\n", planner_status_description));

    //     history.push_str(&format!("Assistant: {}\n", self.current_step_content_raw));
    //     history
    // }
    pub fn total_display_rounds(&self) -> usize {
        self.prev_steps.len()
    }
}
