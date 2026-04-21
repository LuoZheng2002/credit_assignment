use serde::{Deserialize, Serialize};

use crate::multi_agent::session::types::FinalAnswer;

use super::types::{MakeOrChangePlan, NextStepDecision, StepQuality, VerifierComment};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolResponse {
    PythonSuccess(String),
    PythonError(String),
    // Intervention(String),
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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutAction {
    VerifierComment(Option<VerifierComment>),
    PlannerMakeOrChangePlan(Option<MakeOrChangePlan>),
    PlannerDecideNextStep(NextStepDecision),
    PlannerReasoning {
        reasoning: String,
    },
    PlannerToolCall(String),
    ToolCallResponse(ToolResponse),
    PlannerEndStep,
    CompactorCompactStep {
        step_content_compacted: String,
        step_quality: Option<StepQuality>,
    },
    PlannerUpdatePlan(Option<String>),
    // supports both model-provided answer and failed answer
    SubmitFinalAnswer(FinalAnswer),
    // for setting the trajectory state to begin step state, and for marking the end of a step in action_log
    StartNewStep,
    // Special action that force terminates the current step
    SystemInterruptStep(String),
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
            RolloutAction::PlannerReasoning {
                reasoning,
            } => {
                format!("[PlannerReasoning]:\n{}", reasoning)
            }
            RolloutAction::PlannerToolCall(tool_call) => {
                format!("[PlannerToolCall]:\n{}", tool_call)
            }
            RolloutAction::PlannerEndStep => "[PlannerEndStep]".to_string(),
            RolloutAction::CompactorCompactStep {
                step_content_compacted: summary,
                step_quality,
            } => {
                format!(
                    "[PlannerCompactStep]:\n{}\n[StepQuality]: {:?}",
                    summary, step_quality
                )
            }
            RolloutAction::PlannerUpdatePlan(updated_plan) => {
                format!("[PlannerUpdatePlan]:\n{:?}", updated_plan)
            }
            RolloutAction::ToolCallResponse(tool_response) => {
                format!("[ToolCallResponse]:\n{}", tool_response.to_raw_content())
            }
            RolloutAction::VerifierComment(comment) => {
                format!("[VerifierComment]:\n{:?}", comment)
            },
            RolloutAction::SubmitFinalAnswer(final_answer) => {
                format!("[SubmitFinalAnswer]:\n{:?}", final_answer)
            }
            RolloutAction::SystemInterruptStep(reason) => {
                format!("[SystemInterruptStep]:\n{}", reason)
            }
            RolloutAction::StartNewStep => "[StartNewStep]".to_string(),
        }
    }

    pub fn to_concise_string(&self) -> String {
        match self {
            RolloutAction::PlannerMakeOrChangePlan(_plan) => {
                "[PlannerMakeOrChangePlan]".to_string()
            }
            RolloutAction::PlannerDecideNextStep(_mode) => "[PlannerChooseMode]".to_string(),
            RolloutAction::PlannerReasoning{..} => "[PlannerReasoning]".to_string(),
            RolloutAction::PlannerToolCall(_tool_call) => "[PlannerToolCall]".to_string(),
            RolloutAction::PlannerEndStep => "[PlannerEndStep]".to_string(),
            RolloutAction::CompactorCompactStep { .. } => "[PlannerCompactStep]".to_string(),
            RolloutAction::PlannerUpdatePlan(_updated_plan) => "[PlannerUpdatePlan]".to_string(),
            RolloutAction::ToolCallResponse(_tool_response) => "[ToolCallResponse]".to_string(),
            RolloutAction::VerifierComment(_comment) => "[VerifierComment]".to_string(),
            RolloutAction::SubmitFinalAnswer(_final_answer) => "[SubmitFinalAnswer]".to_string(),
            RolloutAction::SystemInterruptStep(_reason) => "[SystemInterruptStep]".to_string(),
            RolloutAction::StartNewStep => "[StartNewStep]".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryActionLog(pub Vec<RolloutAction>);
