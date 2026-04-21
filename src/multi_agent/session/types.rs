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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryStatus {
    VerifierCommenting,
    PlannerChoosingMode {
        verifier_comment: Option<VerifierComment>,
    },
    PlannerMakingOrChangingPlan {
        planner_chosen_mode: NextStepDecision,
        verifier_comment: Option<VerifierComment>,
    },

    PlannerWorkingOnStep {
        planner_chosen_mode: NextStepDecision,
        verifier_comment: Option<VerifierComment>,
        final_answer: Option<String>,
        step_content_raw: String,
    },
    CompactorCompactingStep {
        planner_chosen_mode: NextStepDecision,
        final_answer: Option<String>,
        step_content_raw: String,
        system_interrupted: bool,
    },
    PlannerUpdatingPlan {
        planner_chosen_mode: NextStepDecision,
        final_answer: Option<String>,
        step_content_raw: String,
        step_content_compacted: String,
    },
    StepEnded,
    // The special status that breaks out of the state loop
    SessionEnded {
        final_answer: String,
    },
}

impl TrajectoryStatus {
    pub fn try_get_verifier_comment(&self) -> Option<VerifierComment> {
        match self {
            TrajectoryStatus::PlannerChoosingMode { verifier_comment } => verifier_comment.clone(),
            TrajectoryStatus::PlannerMakingOrChangingPlan {
                verifier_comment, ..
            } => verifier_comment.clone(),
            TrajectoryStatus::PlannerWorkingOnStep {
                verifier_comment, ..
            } => verifier_comment.clone(),
            _ => None,
        }
    }
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
