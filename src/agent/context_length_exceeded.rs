use crate::{
    agent::{
        trajectory_action::TrajectoryAction, trajectory_action_types::FinalAnswer,
        tree_action::TreeAction,
    },
    constants::CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE,
    llm_model::CONTEXT_LENGTH_EXCEEDED_RESPONSE,
    worker_message_tx::log_key_value_pair,
};

pub fn is_context_length_exceeded_response(response: &str) -> bool {
    response == CONTEXT_LENGTH_EXCEEDED_RESPONSE
}

pub fn context_length_exceeded_result(
    question_id: usize,
    final_answer_submitted: bool,
) -> Vec<TreeAction> {
    log_key_value_pair(
        "warning".into(),
        "Model context length exceeded, ending session.".into(),
    );
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
