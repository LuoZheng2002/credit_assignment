use crate::multi_agent::session::SessionState;

fn get_planner_compacting_prompt_before_assistant(session_state: &SessionState) -> String {
    let question = &session_state.question;
    let history_prev_steps = session_state.to_history_prev_steps();
    let current_step_raw_content = session_state.current_step_content_raw.clone();
    format!(
        "\
You are a compactor agent that compacts the content of the current step into a concise summary.\n\
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
Please provide a concise summary of the current step, including what subproblems have been solved, what methods are used, and the results obtained. \n\
If you're sure that the current step has revealed the final answer to the question, mention it in the summary by putting it in \\boxed{{}}.\n\
After the end of the summary, use a json format to include whether the current step has properly utilized the tools, \
whether the current step is a complete step, and whether the current step only focuses on one step in the plan instead of multiple.\n\
For example, if you find in this step, the planner did not use python tool when it could have for improving calculation accuracy, \
the step manages to complete a subgoal specified by the current plan, and the step only solves one subgoal in the plan instead of multiple, then you should output after summary:\n\
{{\"tool\": false, \"complete\": true, \"focused\": true}}\n\n\
Please use the exact JSON format shown above. If the planner did not use tools because there is no way to take advantage of them, you should set \"tool\" to true.\n\
Please provide the summary and the JSON evaluation. Start with \"In this step, \".\
",
        question, history_prev_steps, current_step_raw_content
    )
}

pub fn get_planner_compacting_prompts(session_state: &SessionState) -> (String, String) {
    let prompt_before_assistant = get_planner_compacting_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
