use crate::multi_agent::session::{TrajectoryState, SessionStatus};

fn get_verifier_commenting_prompt_before_assistant(session_state: &TrajectoryState) -> String {
    assert!(matches!(
        session_state.session_status,
        SessionStatus::VerifierCommenting
    ));
    let question = &session_state.question;
    let history_prev_steps = session_state.to_history_prev_steps();
    let last_step_content = session_state
        .prev_steps
        .last()
        .expect("Verifier commenting prompt requires an existing previous step")
        .content_raw
        .clone();
    format!(
        "You are a verifier agent that is trying to evaluate the reasoning steps for the following problem:\n\
<PROBLEM_BEGIN>\n\
{}\n\
<PROBLEM_END>\n\n\
Here is the history of previous steps the planner has done:\n\
<HISTORY_BEGIN>\n\
{}\n\
<HISTORY_END>\n\n\
<CURRENT_STEP_BEGIN>\n\
Planner:\n\
{}\n\
<CURRENT_STEP_END>\n\n\
Your job is to generate the comment for the reasoning of the current step only. You need to provide feedbacks on the following aspects:\n\
1. Are there any calculation or logical mistakes in the current step? If so, point them out in a concise way, do not include too much details.\n\
2. Is the current reasoning direction promising for solving the problem? If not, encourage the planner to try a different direction.\n\
3. Is the length and scope of the current step appropriate? An appropriate step should have a single concrete sub-goal, neither too large nor too small.\n\
4. Does the current step utilize the Python tools when necessary? If the planner attempts to do complex calculations by hand, point it out.\n\
\n\n\
Then at the very end, output a JSON markdown block in this exact format:\n\
```json\n\
{{\"overwrite\": <boolean>, \"change\": <boolean>}}\n\
```\n\
where `overwrite=true` means the current step should be overwritten, and `change=true` means the overall plan should be changed and restarted.\n\
\n\n\
Please start your comment and be concise:\n",
        question, history_prev_steps, last_step_content,
    )
}

pub fn get_verifier_commenting_prompts(session_state: &TrajectoryState) -> (String, String) {
    let prompt_before_assistant = get_verifier_commenting_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
