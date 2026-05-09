use crate::agent::trajectory_action_types::NextStepDecision;
use crate::agent::trajectory_state::TrajectoryState;
use crate::agent::trajectory_status::TrajectoryStatus;

fn get_planner_making_or_changing_plan_status_prompt(change_plan_reason: Option<String>) -> String {
    let header = if let Some(reason) = change_plan_reason {
        format!(
            "Your job is to change the current plan entirely. The reason for changing the plan is:\n{}\n",
            reason
        )
    } else {
        format!("Your job is to make a general plan on how to solve the problem.")
    };
    format!(
        "\
{}\n\
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
Please do not include any reasoning or conclusion in the plan. If you think you can immediately come to a conclusion, use phrases like \"Find if\" or \"Verify if\". \
If later steps depend on some conclusions, make them conditional or \"To be determined\".\n\
Please limit the number of steps to be at most 5. If there are more than 5 steps, mark the 6th step as \"To be determined\".\n\
Please only output the steps.\n\
",
        header
    )
}

pub fn get_planner_making_or_changing_plan_before_assistant(
    session_state: &TrajectoryState<'_>,
) -> String {
    // assert!(matches!(
    //     session_state.status,
    //     TrajectoryStatus::PlannerMakingOrChangingPlan { .. }
    // ));
    let TrajectoryStatus::PlannerMakingOrChangingPlan {
        planner_chosen_mode,
        verifier_comment: _,
    } = &session_state.status
    else {
        panic!("Expected status to be PlannerMakingOrChangingPlan");
    };
    let change_plan_reason = match planner_chosen_mode {
        NextStepDecision::ChangePlan(reason) => Some(reason.clone()),
        _ => None,
    };
    let question = &session_state.question;
    let history_prev_steps = if !session_state.failed_attempts.is_empty() {
        session_state.to_history_prev_steps() + "\n\n"
    } else {
        "".to_string()
    };
    format!(
        "\
You are a planner agent who is responsible for making plans for solving the following problem:\n\
<PROBLEM_BEGIN>\n\
{}\n\
<PROBLEM_END>\n\n\
{}\
{}\
",
        question,
        history_prev_steps,
        get_planner_making_or_changing_plan_status_prompt(change_plan_reason)
    )
}

pub fn get_planner_making_or_changing_plan_prompts(
    session_state: &TrajectoryState<'_>,
) -> (String, String) {
    let prompt_before_assistant =
        get_planner_making_or_changing_plan_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
