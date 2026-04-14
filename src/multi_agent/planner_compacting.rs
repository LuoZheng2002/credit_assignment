use crate::multi_agent::session::SessionState;




fn get_planner_compacting_prompt_before_assistant(session_state: &SessionState) -> String {
    let question = &session_state.question;
    let history_prev_steps = session_state.to_history_prev_steps();
    let current_step_raw_content = session_state.current_step_content_raw.clone();
    format!("\
You are a planner agent that compacts the content of the current step into a concise summary.\n\
The problem to solve is:\n\
<PROBLEM_BEGIN>\n\
{}\n\
<PROBLEM_END>\n\n\
Here is the history of previous steps:\n\
<HISTORY_BEGIN>\n\
{}\n\
<HISTORY_END>\n\n\
The current step is:\n\
{}\n\n\
Please provide a concise summary of the current step, including what subproblems have been solved, what methods are used, and the results obtained. \
Do not include verbose details like tool calls. Please start your summary with \"In this step, \".\
", question, history_prev_steps, current_step_raw_content)
}

pub fn get_planner_compacting_prompts(session_state: &SessionState) -> (String, String) {
    let prompt_before_assistant = get_planner_compacting_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}