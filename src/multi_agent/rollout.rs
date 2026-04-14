use core::panic;

use rand::RngExt;
use reqwest::Client;

use crate::{
    call_llm::call_llm_with_prefix,
    execute_python_code::execute_python_code,
    multi_agent::{
        planner_deciding_next_step::get_planner_deciding_next_step_prompts,
        planner_working_on_step::get_planner_working_on_step_prompts,
        session::{ModelOperation, NextStepDecision, Session, SessionState, SessionStatus},
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
    model_name: &str,
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
        if model_name.to_lowercase().contains("qwen") {
            println!(
                "Warning: tool call does not end with <tool_wait> tag. response: {}",
                response
            );
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

pub async fn execute_planner_tool_call(tool_call: &str) -> String {
    let trimmed_tool_call = tool_call.trim_start();
    assert!(
        trimmed_tool_call.starts_with("```python"),
        "Tool call not properly formatted: {}",
        tool_call
    );
    let Some(fence_end_index) = trimmed_tool_call.rfind("```") else {
        return "```python\nTool call markdown code block not properly closed.\n```".to_string();
    };
    let code_start = trimmed_tool_call
        .find('\n')
        .map(|idx| idx + 1)
        .unwrap_or("```python".len());
    // assert!(
    //     fence_end_index >= code_start,
    //     "Invalid markdown python tool call format"
    // );
    if fence_end_index < code_start {
        return "```python\nTool call markdown code block not properly formatted.\n```".to_string();
    }
    let code = &trimmed_tool_call[code_start..fence_end_index];
    let mut python_code_result = execute_python_code(code.to_string()).await;
    if python_code_result.trim().is_empty() {
        python_code_result = "Python interpreter did not return any output. Please use print statements to retrieve results.".to_string();
    }
    format!("```python\n{}\n```\n", python_code_result.trim())
}

pub fn get_prompt_according_to_session_status(session_state: &SessionState) -> (String, String) {
    match session_state.session_status {
        SessionStatus::PlannerMakingPlan => todo!(),
        SessionStatus::PlannerChoosingMode => get_planner_deciding_next_step_prompts(session_state),
        SessionStatus::PlannerWorkingOnStep => get_planner_working_on_step_prompts(session_state),
        SessionStatus::PlannerCompactingStep => todo!(),
        SessionStatus::PlannerUpdatingPlan => todo!(),
        SessionStatus::VerifierCommenting => get_verifier_commenting_prompts(session_state),
    }
}

pub async fn rollout(
    question_id: usize,
    question: String,
    client: Client,
    model_name: &str,
    verifier_probability: f32,
    rng: &mut impl rand::Rng,
) -> Session {
    // create a state machine
    let mut session = Session::new(question.clone());
    let mut safe_counter = 0; // to prevent infinite loop in case of bugs
    loop {
        let mut session_should_end = false;
        safe_counter += 1;
        if safe_counter > 30 {
            session.session_state.final_answer = Some("The model does not manage to provide a final answer within allowed number of turns.".to_string());
            session_should_end = true;
        }
        let new_operations: Vec<ModelOperation> = match &session.session_state.session_status {
            SessionStatus::PlannerMakingPlan => {
                todo!()
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
                    model_name,
                )
                .await;
                session.add_model_raw_output(response.clone());
                // begin to parse
                let response_json: serde_json::Value = serde_json::from_str(&response.trim())
                    .expect(&format!(
                        "Failed to parse planner choosing mode response as JSON. Response text: {}",
                        response
                    ));
                let choice = response_json["choice"]
                    .as_str()
                    .expect(&format!("Planner choosing mode response JSON does not contain 'choice' field. Response JSON: {}", response_json));
                let chosen_mode = match choice {
                    "continue" => NextStepDecision::Continue,
                    "overwrite_last_step" => {
                        let reason = response_json["reason"]
                            .as_str()
                            .expect(&format!("Planner choosing mode response JSON with 'overwrite_last_step' choice does not contain 'reason' field. Response JSON: {}", response_json));
                        NextStepDecision::OverwriteLastStep(reason.to_string())
                    }
                    "change_plan" => {
                        let fail_reason = response_json["fail_reason"]
                            .as_str()
                            .expect(&format!("Planner choosing mode response JSON with 'change_plan' choice does not contain 'fail_reason' field. Response JSON: {}", response_json));
                        let possible_future_direction = response_json["possible_future_direction"]
                            .as_str()
                            .expect(&format!("Planner choosing mode response JSON with 'change_plan' choice does not contain 'possible_future_direction' field. Response JSON: {}", response_json));
                        NextStepDecision::ChangePlan {
                            fail_reason: fail_reason.to_string(),
                            possible_future_direction: possible_future_direction.to_string(),
                        }
                    }
                    _ => panic!(
                        "Invalid choice field in planner choosing mode response JSON: {}",
                        choice
                    ),
                };
                vec![ModelOperation::PlannerDecideNextStep(chosen_mode)]
            }
            SessionStatus::PlannerWorkingOnStep => {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session.session_state);
                let mut response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model_name,
                )
                .await;
                session.add_model_raw_output(response.clone());
                if response.trim().is_empty() {
                    response += "<end_step>";
                }
                let (reasoning, tool_call) = split_reasoning_and_tool_call(response, model_name);
                let mut operations = Vec::new();
                let mut push_end_step = false;
                if let Some(reasoning) = reasoning {
                    if reasoning.contains("<end_step>") {
                        push_end_step = true;
                    }
                    operations.push(ModelOperation::PlannerReasoning(reasoning));
                }
                if let Some(tool_call) = tool_call {
                    let tool_response = execute_planner_tool_call(&tool_call).await;
                    operations.push(ModelOperation::PlannerToolCall(tool_call));
                    operations.push(ModelOperation::ToolCallResponse(tool_response));
                }
                if push_end_step {
                    operations.push(ModelOperation::PlannerEndStep);
                }
                operations
            }
            SessionStatus::PlannerCompactingStep => {
                todo!()
            }
            SessionStatus::PlannerUpdatingPlan => {
                todo!()
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
                        model_name,
                    )
                    .await;
                    session.add_model_raw_output(response.clone());
                    verifier_comment = Some(response.trim().to_string());
                }
                vec![ModelOperation::VerifierComment(verifier_comment)]
            }
        };

        for operation in new_operations {
            if session.apply_parsed_operation(operation) {
                session_should_end = true;
            }
        }
        println!(
            "[rollout] question index: {}, sub-step: {} finished",
            question_id, safe_counter
        );
        if session_should_end {
            println!(
                "[rollout finishd] question index: {}, total actual rounds: {}, final answer: {}",
                question_id,
                session.session_log.total_actual_rounds(),
                session
                    .session_state
                    .final_answer
                    .as_deref()
                    .unwrap_or("None")
            );
            break;
        }
    }
    session
}
