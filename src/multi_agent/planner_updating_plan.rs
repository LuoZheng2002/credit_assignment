use crate::multi_agent::session::TrajectoryState;

fn get_planner_updating_plan_prompt_before_assistant(session_state: &TrajectoryState<'_>) -> String {
    let question = &session_state.question;
    let history_prev_steps = session_state.to_history_prev_steps();
    let current_step_raw_content = session_state.current_step_content_raw.clone();
    let current_step_summary = session_state
        .current_step_content_compacted
        .clone()
        .expect("When updating the plan, the compacted current step must be available.");
    format!(
        "\
You are a planner agent that updates the current plan based on the progress of the current step.\n\
The problem to solve is:\n\
<PROBLEM_BEGIN>\n\
{}\n\
<PROBLEM_END>\n\n\
Here is the history of previous steps:\n\
<HISTORY_BEGIN>\n\
{}\n\
<HISTORY_END>\n\n\
The current step:\n\
{}\n\
Current step summary:\n\
{}\n\n\
Your need to update the plan based on the following requirements:\n\
1. See if the current step has reached a milestone specified by the current plan. If so, mark the corresponding step in the plan as completed by changing\n\
\"- [ ] Step n: xxx\"\n\
to\n\
\"- [x] Step n: xxx\"\n\
2. If the current step did not reach a milestone but has progress, change the corresponding step in the plan to reflect the progress.\n\
3. If the current step encountered problems, analyze the problems and update the plan to address the problems. You can add new steps, remove existing steps, or change the description of existing steps.\n\
4. If the previous plan is incomplete due to a lack of information, but you have obtained new information from the current step, update the plan to incorporate the new information.\n\
\n\
Please provide the entire updated plan, not just the modified parts, like the following example:\n\
```\n\
- [x] Step 1: xxx\n\
- [ ] Step 2: xxx\n\
- [ ] Step 3: xxx\n\
```\n\
Please only output the steps without the backticks.\n\
",
        question, history_prev_steps, current_step_raw_content, current_step_summary
    )
}

pub fn get_planner_updating_plan_prompts(session_state: &TrajectoryState<'_>) -> (String, String) {
    let prompt_before_assistant = get_planner_updating_plan_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
