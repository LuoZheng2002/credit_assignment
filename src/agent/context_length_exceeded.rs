use crate::{
    agent::{
        trajectory_action::TrajectoryAction, trajectory_action_types::FinalAnswer,
        tree_action::TreeAction,
    },
    call_llm::CONTEXT_LENGTH_EXCEEDED_RESPONSE,
    constants::CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE,
};

pub fn is_context_length_exceeded_response(response: &str) -> bool {
    response == CONTEXT_LENGTH_EXCEEDED_RESPONSE
}

pub fn context_length_exceeded_result(
    question_id: usize,
    final_answer_submitted: bool,
) -> Vec<TreeAction> {
    println!("[Warning] Model context length exceeded, ending session.");
    if !final_answer_submitted {
        vec![TreeAction::AddTrajectoryAction {
            question_id,
            action: TrajectoryAction::SubmitFinalAnswer(FinalAnswer::Failure(
                CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE.to_string(),
            )),
        }]
    } else {
        vec![]
    }
}
