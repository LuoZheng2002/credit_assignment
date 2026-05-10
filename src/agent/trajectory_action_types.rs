use serde::{Deserialize, Serialize};

pub const EMPTY_MESSAGE_HINT: &str = "\
<hint>You are trying to end the step at the start of a step. \
If you have got the answer, put it in \\boxed{} before ending with <end_step>. Otherwise, continue your reasoning in the current step.</hint>";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolResponse {
    PythonSuccess(String),
    PythonError(String),
    // EmptyMessageHint,
}

impl ToolResponse {
    pub fn to_raw_content(&self) -> String {
        match self {
            ToolResponse::PythonSuccess(output) => {
                format!("<tool_response>{}</tool_response>", output)
            }
            ToolResponse::PythonError(error) => {
                format!("<tool_response>Python error: {}</tool_response>", error)
            } // ToolResponse::Intervention(content) => content.clone(),
            // ToolResponse::EmptyMessageHint => EMPTY_MESSAGE_HINT.to_string(),
        }
    }
}

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
    pub current_step_mode: NextStepDecision,
    pub content_raw: String,
    pub content_compacted: String,
}

impl CompletedStep {
    pub fn new(
        current_step_mode: NextStepDecision,
        content_raw: String,
        content_compacted: String,
    ) -> Self {
        Self {
            current_step_mode,
            content_raw,
            content_compacted,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepQuality {
    ProperlyEnded {
        tool: bool,
        complete: bool,
        focused: bool,
    },
    FailedAndAborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierComment {
    pub comment: String,
    pub overwrite: bool,
    pub change_plan: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinalAnswer {
    ModelProvided(String),
    Failure(String),
}

#[derive(Debug, Clone)]
pub struct FailedAttempt {
    pub plan: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifierAndModeSummary {
    VerifierOff,
    VerifierOn,
    VerifierOnAndOverwriteLastStep,
    VerifierOnAndChangePlan,
}
