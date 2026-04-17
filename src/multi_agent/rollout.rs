use core::panic;

use rand::RngExt;
use reqwest::Client;

use crate::{
    call_llm::call_llm_with_prefix,
    deepmath::generate_raw_answers::Model,
    execute_python_code::execute_python_code,
    multi_agent::{
        generate_rollout_answers::RolloutTrajectory,
        planner_compacting::get_planner_compacting_prompts,
        planner_deciding_next_step::get_planner_deciding_next_step_prompts,
        planner_making_plan::get_planner_making_plan_prompts,
        planner_step_continuing::get_planner_step_continuing_prompts,
        planner_step_overwriting::get_planner_step_overwriting_prompts,
        planner_updating_plan::get_planner_updating_plan_prompts,
        session::{
            NextStepDecision, RolloutAction, RolloutActionLogItem, Session, SessionState,
            SessionStatus,
        },
        verifier_commenting::get_verifier_commenting_prompts,
    },
};

#[derive(Debug, Clone, Copy)]
pub enum PlannerState {
    BeginStep,
    MidStep,
}

pub trait ToolCallParser {
    fn start_position(&self, content: &str) -> Option<usize>;
    fn end_position(&self, content: &str, start_position: usize) -> Option<usize>;
}

pub struct MarkdownPythonParser;
impl ToolCallParser for MarkdownPythonParser {
    fn start_position(&self, content: &str) -> Option<usize> {
        content.find("```python")
    }

    fn end_position(&self, content: &str, start_position: usize) -> Option<usize> {
        let after_start = &content[start_position + "```python".len()..];
        if let Some(end_relative) = after_start.find("```") {
            let mut end_position = start_position + "```python".len() + end_relative + "```".len();
            // if there is a '\n' after the closing fence, we also include it in the tool call, as it may be needed for correct formatting when the tool response is inserted back to the planner's reasoning
            if after_start[end_relative + "```".len()..].starts_with('\n') {
                end_position += 1;
            }
            Some(end_position)
        } else {
            None
        }
    }
}

// (Option<String>, Option<String>) means (reasoning, tool_call)
pub fn split_reasoning_and_tool_call(
    response: String,
    model: Model,
) -> (Option<String>, Option<String>) {
    let parsers: Vec<Box<dyn ToolCallParser>> = vec![Box::new(MarkdownPythonParser {})];
    let mut min_start_position = None;
    let mut selected_parser = None;
    for parser in parsers {
        if let Some(start_position) = parser.start_position(&response) {
            if min_start_position.is_none() || start_position < min_start_position.unwrap() {
                min_start_position = Some(start_position);
                selected_parser = Some(parser);
            }
        }
    }
    let Some(start_position) = min_start_position else {
        return (Some(response), None);
    };
    let selected_parser = selected_parser.unwrap();
    let end_position = selected_parser
        .end_position(&response, start_position)
        .unwrap_or(response.len());
    let mut tool_call = response[start_position..end_position].to_string();
    // if after the end position there is immediately a <tool_wait> tag, we also include it in the tool call
    if end_position < response.len() && response[end_position..].trim().starts_with("<tool_wait>") {
        tool_call.push_str("<tool_wait>");
    } else {
        if model.is_qwen() {
            println!("Warning: tool call does not end with <tool_wait> tag.");
        }
        tool_call.push_str("<tool_wait>"); // if there is no <tool_wait> tag, we also add it and trim all the content after the tool call
    }
    let reasoning = if !response[..start_position].trim().is_empty() {
        Some(response[..start_position].to_string()) // do not trim the reasoning part, as leading/trailing spaces may be useful for formatting
    } else {
        None
    };
    (reasoning, Some(tool_call))
}

pub fn extract_content_in_markdown_code_block(content: &str) -> Option<String> {
    let start_fence = "```";
    let end_fence = "```";
    let start_index = content.find(start_fence)?;
    let end_index = content[start_index + start_fence.len()..].find(end_fence)?
        + start_index
        + start_fence.len();
    Some(content[start_index + start_fence.len()..end_index].to_string())
}

pub fn extract_content_in_json_markdown_code_block(content: &str) -> Option<String> {
    let start_fence = "```json";
    let end_fence = "```";
    let start_index = content.find(start_fence)?;
    let end_index = content[start_index + start_fence.len()..].find(end_fence)?
        + start_index
        + start_fence.len();
    Some(content[start_index + start_fence.len()..end_index].to_string())
}

fn get_protocol_value<'a>(response: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{}:", key);
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            return Some(trimmed[prefix.len()..].trim());
        }
    }
    None
}

fn parse_next_step_choice(choice: &str) -> Option<&'static str> {
    let mut has_continue = false;
    let mut has_change = false;
    let mut has_overwrite = false;

    for token in choice
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|token| !token.is_empty())
    {
        let token_lower = token.to_ascii_lowercase();
        match token_lower.as_str() {
            "continue" => has_continue = true,
            "change" => has_change = true,
            "overwrite" | "rewrite" => has_overwrite = true,
            _ => {}
        }
    }

    let matched_count = (has_continue as u8) + (has_change as u8) + (has_overwrite as u8);
    if matched_count != 1 {
        return None;
    }

    if has_continue {
        return Some("continue");
    }
    if has_change {
        return Some("change_plan");
    }
    Some("overwrite_last_step")
}

pub async fn execute_planner_tool_call(tool_call: &str) -> String {
    let trimmed_tool_call = tool_call.trim_start();
    assert!(
        trimmed_tool_call.starts_with("```python"),
        "Tool call not properly formatted: {}",
        tool_call
    );
    let Some(fence_end_index) = trimmed_tool_call.rfind("```") else {
        return "<tool_response>Tool call markdown code block not properly closed.</tool_response>"
            .to_string();
    };
    let code_start = trimmed_tool_call
        .find('\n')
        .map(|idx| idx + 1)
        .unwrap_or("```python".len());
    if fence_end_index < code_start {
        return "<tool_response>Tool call markdown code block not properly formatted.</tool_response>".to_string();
    }
    let code = &trimmed_tool_call[code_start..fence_end_index];
    let mut python_code_result = execute_python_code(code.to_string()).await;
    if python_code_result.trim().is_empty() {
        python_code_result = "Python interpreter did not return any output. Please use print statements to retrieve results.".to_string();
    }
    format!(
        "<tool_response>{}</tool_response>",
        python_code_result.trim()
    )
}

pub fn get_prompt_according_to_session_status(session_state: &SessionState) -> (String, String) {
    match session_state.session_status {
        SessionStatus::PlannerMakingPlan => get_planner_making_plan_prompts(session_state),
        SessionStatus::PlannerChoosingMode => get_planner_deciding_next_step_prompts(session_state),
        SessionStatus::PlannerWorkingOnStep => {
            match session_state
                .planner_chosen_mode
                .as_ref()
                .expect("In PlannerWorkingOnStep status, planner_chosen_mode should be set")
            {
                NextStepDecision::Continue => get_planner_step_continuing_prompts(session_state),
                NextStepDecision::OverwriteLastStep(_) => {
                    get_planner_step_overwriting_prompts(session_state)
                }
                NextStepDecision::ChangePlan(_) => {
                    panic!(
                        "In PlannerWorkingOnStep status, planner_chosen_mode should not be ChangePlan"
                    );
                }
            }
        }
        SessionStatus::PlannerCompactingStep => get_planner_compacting_prompts(session_state),
        SessionStatus::PlannerUpdatingPlan => get_planner_updating_plan_prompts(session_state),
        SessionStatus::VerifierCommenting => get_verifier_commenting_prompts(session_state),
    }
}

pub const SUBMIT_ANSWER_HINT: &str = "\
<hint>It seems you are trying to end the step at the start of the step. \
If you have got the answer, put it in \\boxed{} before ending with <end_step>.</hint>";

// it will output action logs and final trajectory
// it will also load existing logs
pub async fn rollout(
    question_id: usize,
    question: String,
    reference_answer: String,
    loaded_session_log: Vec<RolloutAction>,
    client: Client,
    model: Model,
    verifier_probability: f32,
    rng: &mut impl rand::Rng,
    action_tx: tokio::sync::mpsc::UnboundedSender<RolloutActionLogItem>,
    trajectory_tx: tokio::sync::mpsc::UnboundedSender<RolloutTrajectory>,
) {
    // create a state machine
    let mut session = Session::new(question.clone());
    let mut session_should_end = false;
    for log in loaded_session_log {
        if session.apply_parsed_operation(log) {
            session_should_end = true;
        }
    }
    let mut sub_step_counter = 0; // to prevent infinite loop in case of bugs
    loop {
        if session_should_end {
            println!(
                "[rollout finished] question index: {}, total actual rounds: {}, final answer: {}, correct answer: {}",
                question_id,
                session.session_log.total_actual_rounds(),
                session
                    .session_state
                    .final_answer
                    .as_deref()
                    .unwrap_or("None"),
                reference_answer
            );
            break;
        }
        sub_step_counter += 1;
        if session.session_state.prev_steps.len() > 20 {
            session.session_state.final_answer = Some("The model does not manage to provide a final answer within allowed number of turns.".to_string());
            session_should_end = true;
            println!(
                "[Warning] Number of steps exceeds the limit {}, ending the session.",
                20
            );
        } else if session.session_log.total_actual_rounds() > 150 {
            session.session_state.final_answer = Some("The model does not manage to provide a final answer within allowed number of turns.".to_string());
            session_should_end = true;
            println!(
                "[Warning] Total actual rounds exceeds the limit {}, ending the session.",
                150
            );
        }
        let new_operations: Vec<RolloutAction> = match &session.session_state.session_status {
            SessionStatus::PlannerMakingPlan => {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session.session_state);
                let response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model,
                )
                .await;
                let plan_content = response; // we change to not require the plan to be in a markdown code block
                vec![RolloutAction::PlannerMakePlan(plan_content)]
            }
            SessionStatus::PlannerChoosingMode => {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session.session_state);
                assert_eq!(
                    prompt_after_assistant,
                    String::new(),
                    "Planner deciding next step should not have prompt after assistant"
                );
                let response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model,
                )
                .await;
                let trimmed_response = response.trim();
                let raw_choice = get_protocol_value(trimmed_response, "choice").expect(&format!(
                    "Planner choosing mode response does not contain 'choice: ...' line. Response: {}",
                    trimmed_response
                ));
                let choice = parse_next_step_choice(raw_choice).expect(&format!(
                    "Invalid or ambiguous choice field in planner choosing mode response. Choice must contain exactly one distinct keyword among continue/change/overwrite/rewrite (case-insensitive). choice: {}. Response: {}",
                    raw_choice,
                    trimmed_response
                ));
                let chosen_mode = match choice {
                    "continue" => NextStepDecision::Continue,
                    "overwrite_last_step" => {
                        if session.session_state.can_overwrite_step() {
                            let reason = get_protocol_value(trimmed_response, "reason").expect(
                                &format!(
                                    "Planner choosing mode response with 'overwrite_last_step' choice does not contain 'reason: ...' line. Response: {}",
                                    trimmed_response
                                ),
                            );
                            NextStepDecision::OverwriteLastStep(reason.to_string())
                        } else {
                            println!("[Warning] Overwrite last step is capped.");
                            NextStepDecision::Continue // if cannot overwrite step, we also continue to the next step to avoid getting stuck
                        }
                    }
                    "change_plan" => {
                        if session.session_state.can_change_plan() {
                            let reason = get_protocol_value(trimmed_response, "reason").expect(
                                &format!(
                                    "Planner choosing mode response with 'change_plan' choice does not contain 'reason: ...' line. Response: {}",
                                    trimmed_response
                                ),
                            );
                            NextStepDecision::ChangePlan(reason.to_string())
                        } else {
                            println!("[Warning] Change plan is capped.");
                            NextStepDecision::Continue // if cannot change plan, we also continue to the next step to avoid getting stuck
                        }
                    }
                    _ => panic!(
                        "Invalid choice field in planner choosing mode response JSON: {}",
                        choice
                    ),
                };
                vec![RolloutAction::PlannerDecideNextStep(chosen_mode)]
            }
            SessionStatus::PlannerWorkingOnStep => {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session.session_state);
                let mut response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model,
                )
                .await;
                if response.trim().is_empty() {
                    response = "<end_step>".to_string(); // if the model does not output anything, we treat it as if it outputs <end_step> to prevent getting stuck
                }
                let mut hold_end_step = false;
                if &response == "<end_step>"
                    && &session.session_state.current_step_content_raw == ""
                {
                    println!(
                        "[Warning]: model tries to end the step without providing any content for the step."
                    );
                    // operations.push(ModelOperation::ToolCallResponse(
                    //     SUBMIT_ANSWER_HINT.to_string(),
                    // ));
                    hold_end_step = true;
                }
                let (reasoning, tool_call) = split_reasoning_and_tool_call(response, model);
                let mut push_end_step = false;
                let mut operations = Vec::new();
                if let Some(reasoning) = reasoning {
                    if reasoning.contains("<end_step>") {
                        push_end_step = true;
                    }
                    operations.push(RolloutAction::PlannerReasoning(reasoning));
                }
                if let Some(tool_call) = tool_call {
                    let tool_response = execute_planner_tool_call(&tool_call).await;
                    operations.push(RolloutAction::PlannerToolCall(tool_call));
                    operations.push(RolloutAction::ToolCallResponse(tool_response));
                }
                if hold_end_step {
                    operations.push(RolloutAction::ToolCallResponse(
                        SUBMIT_ANSWER_HINT.to_string(),
                    ));
                }
                let num_additional_actions_allowed = session
                    .session_state
                    .num_additional_actions_allowed_in_current_step();
                if operations.len() > num_additional_actions_allowed {
                    println!(
                        "[Warning] Number of actions in the current step {} exceeds the limit {}. Only the first {} actions will be applied.",
                        operations.len(),
                        num_additional_actions_allowed,
                        num_additional_actions_allowed
                    );
                    operations.truncate(num_additional_actions_allowed);
                }
                if push_end_step && !hold_end_step {
                    operations.push(RolloutAction::PlannerEndStep);
                }
                operations
            }
            SessionStatus::PlannerCompactingStep => {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session.session_state);
                let response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model,
                )
                .await;
                vec![RolloutAction::PlannerCompactStep(response)]
            }
            SessionStatus::PlannerUpdatingPlan => {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session.session_state);
                let response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model,
                )
                .await;
                let updated_plan_content = response; // we change to not require the updated plan to be in a markdown code block
                vec![RolloutAction::PlannerUpdatePlan(updated_plan_content)]
            }
            SessionStatus::VerifierCommenting => {
                let mut verifier_comment = None;
                if rng.random::<f32>() <= verifier_probability {
                    // let prompt_before_assistant =
                    //     get_verifier_commenting_prompt_before_assistant(&session.session_state);
                    let (prompt_before_assistant, prompt_after_assistant) =
                        get_prompt_according_to_session_status(&session.session_state);
                    assert_eq!(
                        prompt_after_assistant,
                        String::new(),
                        "Verifier commenting should not have prompt after assistant"
                    );
                    let response = call_llm_with_prefix(
                        client.clone(),
                        prompt_before_assistant,
                        prompt_after_assistant,
                        model,
                    )
                    .await;
                    verifier_comment = Some(response.trim().to_string());
                }
                vec![RolloutAction::VerifierComment(verifier_comment)]
            }
        };

        for operation in new_operations {
            if session.apply_parsed_operation(operation.clone()) {
                session_should_end = true;
            }
            // log the action
            let log_item = RolloutActionLogItem {
                question_id,
                action: operation,
            };
            action_tx.send(log_item).unwrap();
        }
        println!(
            "[rollout] question index: {}, sub-step: {} finished, num prev steps: {}",
            question_id,
            sub_step_counter,
            session.session_state.prev_steps.len()
        );
    }
    let rollout_trajectory = RolloutTrajectory {
        id: question_id,
        question,
        model_answer: session
            .session_state
            .final_answer
            .clone()
            .unwrap_or("No answer found".into()),
        correct_answer: reference_answer, // we will fill in the correct answer later when we evaluate the trajectory, to avoid data leakage
        trajectory: session.session_log,
    };
    trajectory_tx.send(rollout_trajectory).unwrap();
}
