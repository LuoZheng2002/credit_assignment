use crate::multi_agent::session::{SessionState, SessionStatus};

pub fn get_planner_prompt_before_assistant(
    question: &str,
    history_prev_steps: &str,
    planner_status_prompt: String,
) -> String {
    format!(
        "You are a planner agent that is trying to solve the following problem step by step:\n\
<PROBLEM_BEGIN>\n\
{}\n\
<PROBLEM_END>\n\n\
Here is the history of previous steps you have done:\n\
<HISTORY_BEGIN>\n\
{}\n\
<HISTORY_END>\n\n\
{}",
        question, history_prev_steps, planner_status_prompt
    )
}

fn get_planner_deciding_next_step_status_prompt() -> String {
    "\
A new step is just about to begin. Your job is to determine the direction of the new step based on the history. You have the following choices:\n\
1. Continue: You are confident about the current plan and want to proceed with it.\n\
2. Overwrite Last Step: You find the last step problematic, and want to rewrite it while sticking to the current plan.\n\
3. Change Plan: You find the current plan is not promising given the history of previous steps, and want to start over with a new plan.\n\
\n\
If you choose \"Continue\", write only the following json: {\"choice\": \"continue\"}.\n\
If you choose \"Overwrite Last Step\", write only the following json, with contents in the square brackets replaced with appropriate contents: \
{\"choice\": \"overwrite_last_step\", \"reason\": \"[Your reason here, including what goes wrong in the last step and what can be improved in the new iteration.]\"}.\n\
If you choose \"Change Plan\", write only the following json, with contents in the square brackets replaced with appropriate contents: \
{\"choice\": \"change_plan\", \"fail_reason\": \"[Your reason here, describing why the current plan is not promising.]\", \
\"possible_future_direction\": \"[Briefly describe a possible direction without elaborating on detailed plans.]\"}.\n\
Please be careful with the json character escape rule if you try to include math formula.".to_string()
}

fn get_planner_deciding_next_step_prompt_before_assistant(session_state: &SessionState) -> String {
    assert!(matches!(
        session_state.session_status,
        SessionStatus::PlannerChoosingMode
    ));
    let question = &session_state.question;
    let planner_status_prompt = get_planner_deciding_next_step_status_prompt();
    let history_prev_steps = session_state.to_history_prev_steps();
    get_planner_prompt_before_assistant(question, &history_prev_steps, planner_status_prompt)
}

pub fn get_planner_deciding_next_step_prompts(session_state: &SessionState) -> (String, String) {
    let prompt_before_assistant =
        get_planner_deciding_next_step_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
