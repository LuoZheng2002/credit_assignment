use crate::multi_agent::session::{SessionState, SessionStatus};

fn get_planner_making_plan_status_prompt() -> String {
    "\
Your job is to make a general plan on how to solve the problem. \
You need to be explicit about what to do on each step and what each step should achieve.\n\
Please write the plan as a list of steps in the same format as the following example:\n\
```\n\
- [ ] Step 1: xxx.\n\
- [ ] Step 2: xxx.\n\
- [ ] Step 3: xxx.\n\
```\n\
Each step should only span one line.\n\
If it is unclear what to do in later steps, or it depends on the result of previous steps, you can write \"To be determined\" for the last step.\n\
For example:\n\
```\n\
- [ ] Step 1: xxx.\n\
- [ ] Step 2: To be determined.\n\
```\n\
Please do not include any conclusion in the plan. If you think you can immediately come to a conclusion, use phrases like \"Find if\" or \"Verify if\". \
If later steps depend on some conclusions, make them conditional or \"To be determined\".\n\
Please put your final plan within the markdown code block with triple backticks.\n\
"
    .to_string()
}

pub fn get_planner_making_plan_before_assistant(session_state: &SessionState) -> String {
    assert!(matches!(
        session_state.session_status,
        SessionStatus::PlannerMakingPlan
    ));
    let question = &session_state.question;
    let history_prev_steps = if !session_state.failed_attempts.is_empty() {
        session_state.to_history_prev_steps() + "\n\n"
    } else {
        "".to_string()
    };
    format!(
        "\
You are a planner agent that makes a plan to solve the following problem step by step:\n\
<PROBLEM_BEGIN>\n\
{}\n\
<PROBLEM_END>\n\n\
{}\
{}\
",
        question,
        history_prev_steps,
        get_planner_making_plan_status_prompt()
    )
}

pub fn get_planner_making_plan_prompts(session_state: &SessionState) -> (String, String) {
    let prompt_before_assistant = get_planner_making_plan_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
