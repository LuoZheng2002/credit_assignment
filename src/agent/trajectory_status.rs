use crate::agent::trajectory_action_types::{FinalAnswer, NextStepDecision, VerifierComment};

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
        step_content_raw: String,
    },
    CompactorCompactingStep {
        planner_chosen_mode: NextStepDecision,
        step_content_raw: String,
    },
    PlannerUpdatingPlan {
        planner_chosen_mode: NextStepDecision,
        step_content_raw: String,
        step_content_compacted: String,
    },
    StepEnded,
    // The special status that breaks out of the state loop
    SessionEnded {
        final_answer: FinalAnswer,
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
