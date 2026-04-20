use crate::multi_agent::session::{TrajectoryState, TrajectoryStatus};

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
Output format requirement (strict): each key-value pair must occupy exactly one line using `key: value`.\n\
Do not output JSON. Do not output markdown code block.\n\
\n\
If you choose \"Continue\", output exactly:\n\
choice: Continue\n\
\n\
If you choose \"Overwrite Last Step\", output exactly two lines:\n\
choice: Overwrite Last Step\n\
reason: [Your reason here, including what goes wrong in the last step and what can be improved in the new iteration.]\n\
\n\
If you choose \"Change Plan\", output exactly two lines:\n\
choice: Change Plan\n\
reason: [Describe both why the current plan is not promising and what possible future direction you suggest, in one single-line sentence.]".to_string()
}

fn get_planner_deciding_next_step_prompt_before_assistant(session_state: &TrajectoryState<'_>) -> String {
    assert!(matches!(
        session_state.status,
        TrajectoryStatus::PlannerChoosingMode
    ));
    let question = &session_state.question;
    let planner_status_prompt = get_planner_deciding_next_step_status_prompt();
    let history_prev_steps = session_state.to_history_prev_steps();
    get_planner_prompt_before_assistant(question, &history_prev_steps, planner_status_prompt)
}

pub fn get_planner_deciding_next_step_prompts(session_state: &TrajectoryState<'_>) -> (String, String) {
    let prompt_before_assistant =
        get_planner_deciding_next_step_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
