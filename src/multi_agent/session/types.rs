use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryStatus {
    PlannerMakingOrChangingPlan,
    PlannerKeepingCurrentPlan,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifierAndModeSummary {
    VerifierOff,
    VerifierOn,
    VerifierOnAndOverwriteLastStep,
    VerifierOnAndChangePlan,
}
