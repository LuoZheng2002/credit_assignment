use crate::multi_agent::session::{SessionState, SessionStatus};

fn get_verifier_commenting_prompt_before_assistant(session_state: &SessionState) -> String {
    assert!(matches!(
        session_state.session_status,
        SessionStatus::VerifierCommenting
    ));
    let question = &session_state.question;
    let history_prev_steps = session_state.to_history_prev_steps();
    let current_step_content = session_state.current_step_content_raw.clone();
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
1. Are there any calculation or logical mistakes in the current step? If so, point them out. If there are values produced, verify if they satisfy the problem requirements by plugging them in.\n\
2. Is the current reasoning direction promising for solving the problem? If not, encourage the planner to try a different direction.\n\
3. Is the length and scope of the current step appropriate? An appropriate step should have a single concrete sub-goal, neither too large nor too small. If you think the current step scope is inappropriate, make suggestions on how to plan the steps better.\n\
4. Does the current step utilize the Python tools when necessary? If the planner attempts to do complex calculations by hand, point it out. \
Are there opportunities for the planner to use Python for complex but pervasive problems like solving a set of linear equations? Encourage the planner to leverage these tools even when calculations appear to be manageable.\n\
\n\n\
Please start your comment and be concise:\n",
        question, history_prev_steps, current_step_content,
    )
}

pub fn get_verifier_commenting_prompts(session_state: &SessionState) -> (String, String) {
    let prompt_before_assistant = get_verifier_commenting_prompt_before_assistant(session_state);
    (prompt_before_assistant, String::new())
}
