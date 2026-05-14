use crate::agent::trajectory_state::TrajectoryState;
use crate::agent::trajectory_status::TrajectoryStatus;

fn get_planner_compacting_prompt_before_assistant(
    trajectory_state: &TrajectoryState<'_>,
) -> String {
    let question = &trajectory_state.question;
    let history_prev_steps = trajectory_state.to_history_prev_steps();
    let TrajectoryStatus::CompactorCompactingStep {
        planner_chosen_mode: _,
        step_content_raw,
    } = &trajectory_state.status
    else {
        panic!(
            "TrajectoryStatus must be CompactorCompactingStep when calling get_planner_compacting_prompt_before_assistant"
        );
    };
    // let current_step_raw_content = trajectory_state.current_step_content_raw.clone();
    let current_step_raw_content = step_content_raw.clone();
    format!(
        "\
You are a compactor agent that compacts the content of the current step into a CONCISE summary.\n\
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
Please provide a CONCISE summary of the current step, which may include what subproblems have been solved, what methods are used, and the results obtained. \n\
If you're sure that the current step has revealed the final answer to the question, mention it in the summary by putting it in \\boxed{{}}.\n\
After the end of the summary, output a JSON evaluation in a markdown code fence with language tag json to include whether the current step has properly utilized the tools, \
whether the current step is a complete step, and whether the current step only focuses on one step in the plan instead of multiple.\n\
A sample json content after summary can be:\n\
```json\n\
{{\"tool\": true, \"complete\": true, \"focused\": true}}\n\
```\n\n\
Please use the exact JSON format shown above and keep the json fenced block as the last part of your output, with nothing after it. If the planner did not use tools because there is no way to take advantage of them, you should set \"tool\" to true.\n\
Please provide the CONCISE summary and the JSON evaluation. Start with \"In this step, \".\
",
        question, history_prev_steps, current_step_raw_content
    )
}

pub fn get_planner_compacting_prompts(session_state: &TrajectoryState<'_>) -> (String, String) {
    let prompt_before_assistant = get_planner_compacting_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
