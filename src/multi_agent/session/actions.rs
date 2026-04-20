use serde::{Deserialize, Serialize};

use super::types::{MakeOrChangePlan, NextStepDecision, StepQuality, VerifierComment};

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
    PlannerMakeOrChangePlan(Option<MakeOrChangePlan>),
    PlannerDecideNextStep(NextStepDecision),
    PlannerReasoning(String),
    PlannerToolCall(String),
    ToolCallResponse(ToolResponse),
    PlannerEndStep,
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
                "[PlannerMakeOrChangePlan]".to_string()
            }
            RolloutAction::PlannerDecideNextStep(_mode) => {
                "[PlannerChooseMode]".to_string()
            }
            RolloutAction::PlannerReasoning(_reasoning) => "[PlannerReasoning]".to_string(),
            RolloutAction::PlannerToolCall(_tool_call) => "[PlannerToolCall]".to_string(),
            RolloutAction::PlannerEndStep => "[PlannerEndStep]".to_string(),
            RolloutAction::PlannerCompactStep { .. } => "[PlannerCompactStep]".to_string(),
            RolloutAction::PlannerUpdatePlan(_updated_plan) => "[PlannerUpdatePlan]".to_string(),
            RolloutAction::ToolCallResponse(_tool_response) => "[ToolCallResponse]".to_string(),
            RolloutAction::VerifierComment(_comment) => "[VerifierComment]".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryActionLog(pub Vec<RolloutAction>);
