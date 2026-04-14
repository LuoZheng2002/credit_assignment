use crate::multi_agent::{planner_deciding_next_step::get_planner_prompt_before_assistant, session::{NextStepDecision, SessionState, SessionStatus}};

pub fn get_step_mode_prompt(is_overwriting: bool, is_planner: bool) -> String {
    let subject = if is_planner {
        "You are"
    } else {
        "The planner is"
    };
    match is_overwriting {
        false => format!(
            "{} willing to work on the next step based on the results of previous steps.",
            subject
        ),
        true => format!(
            "{} willing to overwrite the last step with the current step.",
            subject
        ),
    }
}


fn get_planner_working_on_step_status_prompt(step_mode: NextStepDecision) -> String {
    let tool_prompt: String = "\
You can both reason in plain texts and use the following tools in this step:\n\
1. Python code executor: You're encouraged to use it for calculations to ensure correctness. You can invoke python code by outputting a markdown Python code block.\n\
IMPORTANT: always use Python's print statement to output the result, otherwise the result will not be shown.\n\
IMPORTANT: after calling any tool, immediately output a <tool_wait> to obtain the tool's response.\n\
\n\
If the current step's goal is achieved, end your response with <end_step>. DO NOT start the next step in the same turn.\n\
If you have got the final answer to submit, put the answer in \\boxed{} in a concise form. \
Do not put anything else other than the final answer in \\boxed{}.".to_string();
    let is_overwriting = step_mode.is_overwriting();
    let step_mode_prompt = get_step_mode_prompt(is_overwriting, true);
    format!(
        "\
The verifier's comment is only for reference and may not be true. \
Please do not explicitly quote the verifier or try to respond to it in your reasoning.\n\
{}\n\
\n\
{}\n\
Please identify the next step from the current plan and work on it. Only work on one step before outputting <enqd_step>.\n\
Begin your new step:",
        tool_prompt, step_mode_prompt
    )
}

fn get_planner_working_on_step_prompt_before_assistant(session_state: &SessionState) -> String {
    assert!(matches!(
        session_state.session_status,
        SessionStatus::PlannerWorkingOnStep
    ));
    let question = &session_state.question;
    let chosen_mode = session_state
        .planner_chosen_mode
        .clone()
        .expect("Planner chosen mode should be set when session status is not PlannerChoosingMode");
    let planner_status_prompt = get_planner_working_on_step_status_prompt(chosen_mode);
    let history_prev_steps = session_state.to_history_prev_steps();
    get_planner_prompt_before_assistant(question, planner_status_prompt, &history_prev_steps)
}

fn get_planner_prompt_after_assistant(session_state: &SessionState) -> String {
    // only working on step needs the prompt after assistant
    assert!(matches!(
        session_state.session_status,
        SessionStatus::PlannerWorkingOnStep
    ));
    session_state.current_step_content_raw.clone()
}

pub fn get_planner_working_on_step_prompts(session_state: &SessionState) -> (String, String) {
    let prompt_before_assistant = get_planner_working_on_step_prompt_before_assistant(session_state);
    let prompt_after_assistant = get_planner_prompt_after_assistant(session_state);
    (prompt_before_assistant, prompt_after_assistant)
}