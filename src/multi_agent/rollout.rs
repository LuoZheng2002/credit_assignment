use core::panic;

use rand::RngExt;
use reqwest::Client;

use crate::{
    call_llm::call_llm_with_prefix,
    deepmath::generate_raw_answers::Model,
    execute_python_code::execute_python_code,
    multi_agent::{
        planner_compacting::get_planner_compacting_prompts,
        planner_deciding_next_step::get_planner_deciding_next_step_prompts,
        planner_making_plan::get_planner_making_plan_prompts,
        planner_updating_plan::get_planner_updating_plan_prompts,
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
        SessionStatus::PlannerMakingPlan => get_planner_making_plan_prompts(session_state),
        SessionStatus::PlannerChoosingMode => get_planner_deciding_next_step_prompts(session_state),
        SessionStatus::PlannerWorkingOnStep => get_planner_working_on_step_prompts(session_state),
        SessionStatus::PlannerCompactingStep => get_planner_compacting_prompts(session_state),
        SessionStatus::PlannerUpdatingPlan => get_planner_updating_plan_prompts(session_state),
        SessionStatus::VerifierCommenting => get_verifier_commenting_prompts(session_state),
    }
}

pub async fn rollout(
    question_id: usize,
    question: String,
    client: Client,
    model: Model,
    verifier_probability: f32,
    rng: &mut impl rand::Rng,
) -> Session {
    // create a state machine
    let mut session = Session::new(question.clone());
    let mut sub_step_counter = 0; // to prevent infinite loop in case of bugs
    loop {
        let mut session_should_end = false;
        sub_step_counter += 1;
        if session.session_state.prev_steps.len() > 20 || sub_step_counter > 150 {
            session.session_state.final_answer = Some("The model does not manage to provide a final answer within allowed number of turns.".to_string());
            session_should_end = true;
        }
        let new_operations: Vec<ModelOperation> = match &session.session_state.session_status {
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
                session.add_model_raw_output(response.clone());
                // extract the content in the markdown code block
                let plan_content = extract_content_in_markdown_code_block(&response).expect(&format!(
                    "Failed to extract markdown code block content for planner making plan. Response: {}",
                    response
                ));
                vec![ModelOperation::PlannerMakePlan(plan_content)]
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
                session.add_model_raw_output(response.clone());
                // begin to parse
                // let response_json: serde_json::Value = serde_json::from_str(&response.trim())
                //     .expect(&format!(
                //         "Failed to parse planner choosing mode response as JSON. Response text: {}",
                //         response
                //     ));
                let response_json: serde_json::Value = match serde_json::from_str(&response.trim())
                {
                    Ok(json) => json,
                    Err(e) => {
                        // call llm to fix the Json format error, only outputs the fixed json
                        let fix_json_prompt = format!(
                            "The following JSON is not properly formatted and cannot be parsed. Please fix the JSON format error and only output the fixed JSON without any explanation. JSON: {}. Error: {} You should start immediately with open curly bracket and end with close curly bracket.",
                            response.trim(),
                            e
                        );
                        let fixed_response = call_llm_with_prefix(
                            client.clone(),
                            fix_json_prompt,
                            String::new(),
                            Model::Gpt4o,
                        )
                        .await;
                        session.add_model_raw_output(fixed_response.clone());
                        // serde_json::from_str(&fixed_response.trim()).expect(&format!(
                        //     "Failed to parse fixed planner choosing mode response as JSON. Fixed response text: {}",
                        //     fixed_response
                        // ))
                        match serde_json::from_str(&fixed_response.trim()) {
                            Ok(json) => json,
                            Err(e) => {
                                // extract the content in the markdown code block of the fixed response and try to parse it as JSON again, as the llm may output the fixed JSON in a markdown code block
                                if let Some(content) =
                                    extract_content_in_json_markdown_code_block(&fixed_response)
                                {
                                    serde_json::from_str(&content.trim()).expect(&format!(
                                        "Failed to parse JSON content in markdown code block of fixed planner choosing mode response. Content: {}. Original fixed response: {}. Error: {}",
                                        content,
                                        fixed_response,
                                        e
                                    ))
                                } else {
                                    panic!(
                                        "Failed to parse fixed planner choosing mode response as JSON and could not extract JSON content from markdown code block. Fixed response: {}. Error: {}",
                                        fixed_response, e
                                    );
                                }
                            }
                        }
                    }
                };
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
                    model,
                )
                .await;
                session.add_model_raw_output(response.clone());
                if response.trim().is_empty() {
                    response += "<end_step>";
                }
                let (reasoning, tool_call) = split_reasoning_and_tool_call(response, model);
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
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(&session.session_state);
                let response = call_llm_with_prefix(
                    client.clone(),
                    prompt_before_assistant,
                    prompt_after_assistant,
                    model,
                )
                .await;
                session.add_model_raw_output(response.clone());
                vec![ModelOperation::PlannerCompactStep(response)]
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
                session.add_model_raw_output(response.clone());
                // extract the content in the markdown code block
                let updated_plan_content = extract_content_in_markdown_code_block(&response).expect(&format!(
                    "Failed to extract markdown code block content for planner updating plan. Response: {}",
                    response
                ));
                vec![ModelOperation::PlannerUpdatePlan(updated_plan_content)]
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
            "[rollout] question index: {}, sub-step: {} finished, num prev steps: {}",
            question_id,
            sub_step_counter,
            session.session_state.prev_steps.len()
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
