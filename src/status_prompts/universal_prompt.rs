use crate::{
    agent::{
        trajectory_action_types::NextStepDecision, trajectory_state::TrajectoryState,
        trajectory_status::TrajectoryStatus,
    },
    status_prompts::{
        planner_compacting::get_planner_compacting_prompts,
        planner_deciding_next_step::get_planner_deciding_next_step_prompts,
        planner_making_or_changing_plan::get_planner_making_or_changing_plan_prompts,
        planner_step_continuing::get_planner_step_continuing_prompts,
        planner_step_overwriting::get_planner_step_overwriting_prompts,
        planner_updating_plan::get_planner_updating_plan_prompts,
        verifier_commenting::get_verifier_commenting_prompts,
    },
};

pub fn get_prompt_according_to_session_status(
    session_state: &TrajectoryState<'_>,
    // status: &TrajectoryStatus,
) -> (String, String) {
    match &session_state.status {
        TrajectoryStatus::PlannerMakingOrChangingPlan { .. } => {
            get_planner_making_or_changing_plan_prompts(session_state)
        }
        TrajectoryStatus::PlannerChoosingMode { .. } => {
            get_planner_deciding_next_step_prompts(session_state)
        }
        TrajectoryStatus::PlannerWorkingOnStep {
            planner_chosen_mode,
            ..
        } => match planner_chosen_mode {
            NextStepDecision::Continue => get_planner_step_continuing_prompts(session_state),
            NextStepDecision::OverwriteLastStep(_) => {
                get_planner_step_overwriting_prompts(session_state)
            }
            NextStepDecision::ChangePlan(_) => get_planner_step_continuing_prompts(session_state),
        },
        TrajectoryStatus::CompactorCompactingStep { .. } => {
            get_planner_compacting_prompts(session_state)
        }
        TrajectoryStatus::PlannerUpdatingPlan { .. } => {
            get_planner_updating_plan_prompts(session_state)
        }
        TrajectoryStatus::VerifierCommenting => get_verifier_commenting_prompts(session_state),
        // TrajectoryStatus::Empty        | 
        TrajectoryStatus::StepEnded
        | TrajectoryStatus::TrajectoryEnded { .. } => (String::new(), String::new()),
    }
}
