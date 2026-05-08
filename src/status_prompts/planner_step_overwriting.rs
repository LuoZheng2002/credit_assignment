use crate::{
    multi_agent::session::{NextStepDecision, TrajectoryState, TrajectoryStatus},
    status_prompts::{
        planner_deciding_next_step::get_planner_prompt_before_assistant,
        planner_step_continuing::{
            STEP_HALT_PROMPT, TOOL_PROMPT, get_planner_prompt_after_assistant,
        },
    },
};

fn get_planner_step_overwriting_status_prompt(step_mode: NextStepDecision) -> String {
    let NextStepDecision::OverwriteLastStep(overwrite_reason) = step_mode else {
        panic!("Only overwriting mode should call get_planner_step_overwriting_status_prompt");
    };
    format!(
        "\
The verifier's comment is only for reference. \
Please do not explicitly quote the verifier or try to respond to it.\n\
{}\n\
{}\n\
\n\
Your current task is to rewrite the last step instead of starting a new one.\n\
The reason for rewriting is the following:\n\
{}\n\n\
Begin your step:",
        TOOL_PROMPT, STEP_HALT_PROMPT, overwrite_reason
    )
}

fn get_planner_step_overwriting_prompt_before_assistant(
    session_state: &TrajectoryState<'_>,
) -> String {
    // assert!(matches!(
    //     session_state.status,
    //     TrajectoryStatus::PlannerWorkingOnStep
    // ));
    let TrajectoryStatus::PlannerWorkingOnStep {
        planner_chosen_mode,
        ..
    } = &session_state.status
    else {
        panic!(
            "TrajectoryStatus must be PlannerWorkingOnStep when calling get_planner_step_overwriting_prompt_before_assistant"
        );
    };
    let question = &session_state.question;
    let chosen_mode = planner_chosen_mode.clone();
    let planner_status_prompt = get_planner_step_overwriting_status_prompt(chosen_mode);
    let history_prev_steps = session_state.to_history_prev_steps();
    get_planner_prompt_before_assistant(question, &history_prev_steps, planner_status_prompt)
}

pub fn get_planner_step_overwriting_prompts(
    session_state: &TrajectoryState<'_>,
) -> (String, String) {
    let prompt_before_assistant =
        get_planner_step_overwriting_prompt_before_assistant(session_state);
    let prompt_after_assistant = get_planner_prompt_after_assistant(session_state);
    (prompt_before_assistant, prompt_after_assistant)
}
