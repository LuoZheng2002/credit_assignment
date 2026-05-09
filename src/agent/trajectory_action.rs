use serde::{Deserialize, Serialize};

use crate::agent::trajectory_action_types::{FinalAnswer, ToolResponse};

use super::trajectory_action_types::{
    MakeOrChangePlan, NextStepDecision, StepQuality, VerifierComment,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrajectoryAction {
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
    // for splitting leaf registration from branching-node selection after a trajectory ends
    StartDeterminingBranchingNode,
    // Special action that force terminates the current step
    SystemInterrupt(String),
}

impl TrajectoryAction {
    pub fn to_pretty_string(&self) -> String {
        match self {
            TrajectoryAction::PlannerMakeOrChangePlan(plan) => {
                format!("[PlannerMakeOrChangePlan]:\n{:?}", plan)
            }
            TrajectoryAction::PlannerDecideNextStep(mode) => {
                format!("[PlannerChooseMode]: {:?}", mode)
            }
            TrajectoryAction::PlannerReasoning { reasoning } => {
                format!("[PlannerReasoning]:\n{}", reasoning)
            }
            TrajectoryAction::PlannerToolCall(tool_call) => {
                format!("[PlannerToolCall]:\n{}", tool_call)
            }
            TrajectoryAction::PlannerEndStep => "[PlannerEndStep]".to_string(),
            TrajectoryAction::CompactorCompactStep {
                step_content_compacted: summary,
                step_quality,
            } => {
                format!(
                    "[PlannerCompactStep]:\n{}\n[StepQuality]: {:?}",
                    summary, step_quality
                )
            }
            TrajectoryAction::PlannerUpdatePlan(updated_plan) => {
                format!("[PlannerUpdatePlan]:\n{:?}", updated_plan)
            }
            TrajectoryAction::ToolCallResponse(tool_response) => {
                format!("[ToolCallResponse]:\n{}", tool_response.to_raw_content())
            }
            TrajectoryAction::VerifierComment(comment) => {
                format!("[VerifierComment]:\n{:?}", comment)
            }
            TrajectoryAction::SubmitFinalAnswer(final_answer) => {
                format!("[SubmitFinalAnswer]:\n{:?}", final_answer)
            }
            TrajectoryAction::SystemInterrupt(reason) => {
                format!("[SystemInterruptStep]:\n{}", reason)
            }
            TrajectoryAction::StartNewStep => "[StartNewStep]".to_string(),
            TrajectoryAction::StartDeterminingBranchingNode => {
                "[StartDeterminingBranchingNode]".to_string()
            }
        }
    }

    pub fn to_concise_string(&self) -> String {
        match self {
            TrajectoryAction::PlannerMakeOrChangePlan(_plan) => {
                "[PlannerMakeOrChangePlan]".to_string()
            }
            TrajectoryAction::PlannerDecideNextStep(_mode) => "[PlannerChooseMode]".to_string(),
            TrajectoryAction::PlannerReasoning { .. } => "[PlannerReasoning]".to_string(),
            TrajectoryAction::PlannerToolCall(_tool_call) => "[PlannerToolCall]".to_string(),
            TrajectoryAction::PlannerEndStep => "[PlannerEndStep]".to_string(),
            TrajectoryAction::CompactorCompactStep { .. } => "[PlannerCompactStep]".to_string(),
            TrajectoryAction::PlannerUpdatePlan(_updated_plan) => "[PlannerUpdatePlan]".to_string(),
            TrajectoryAction::ToolCallResponse(_tool_response) => "[ToolCallResponse]".to_string(),
            TrajectoryAction::VerifierComment(_comment) => "[VerifierComment]".to_string(),
            TrajectoryAction::SubmitFinalAnswer(_final_answer) => "[SubmitFinalAnswer]".to_string(),
            TrajectoryAction::SystemInterrupt(_reason) => "[SystemInterruptStep]".to_string(),
            TrajectoryAction::StartNewStep => "[StartNewStep]".to_string(),
            TrajectoryAction::StartDeterminingBranchingNode => {
                "[StartDeterminingBranchingNode]".to_string()
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryActionLog(pub Vec<TrajectoryAction>);
