use core::panic;

use rand::RngExt;
use reqwest::Client;

use crate::{
    call_llm::call_llm,
    execute_python_code::execute_python_code,
    multi_agent::session::{
        ActualStepMode, ModelOperation, PlannerStatus, Session, SessionStatus, StepDirection,
    },
};

#[derive(Debug, Clone, Copy)]
pub enum PlannerState {
    BeginStep,
    MidStep,
}

fn get_planner_status_prompt(planner_status: PlannerStatus) -> String {
    match planner_status {
        PlannerStatus::PlannerChoosingMode => {
            "Currently a new step is just about to begin. Your job is to determine the mode of this step. You have the following choices:\n\
1. PROCEED: You are confident about the current reasoning direction and want to proceed with it.\n\
2. CHANGE_PLAN: You find the previous steps not leading to a good direction, and want to change the plan and try a different reasoning direction.\n\
3. OVERWRITE_LAST_STEP_AND_PROCEED: You find the last step problematic, and want to rewrite it while maintaining the current reasoning direction.\n\
4. OVERWRITE_LAST_STEP_AND_CHANGE_PLAN: You find the last step problematic, and want to rewrite it and also change the reasoning direction.\n\
5. COMPACT: You find the context length too long, and want to compact the previous steps into a more concise form, while maintaining the current reasoning direction.\n\
6. SUBMIT_ANSWER: You think you have reached the end of reasoning and want to begin reporting the final answer.\n\n\
Please output exactly one of PROCEED, CHANGE_PLAN, OVERWRITE_LAST_STEP_AND_PROCEED, OVERWRITE_LAST_STEP_AND_CHANGE_PLAN, COMPACT, SUBMIT_ANSWER and nothing else.".to_string()
        }
        PlannerStatus::PlannerChosen(step_mode) => {
            let step_mode_prompt: String = match step_mode {
                ActualStepMode::Append(step_direction) => {
                    match step_direction {
                        StepDirection::Proceed => "You are willing to proceed with the current reasoning direction.".to_string(),
                        StepDirection::ChangePlan => "You are attempting a different reasoning direction from the previous steps.".to_string(),
                    }
                },
                ActualStepMode::OverwriteLastStep(step_direction) => {
                    match step_direction {
                        StepDirection::Proceed => "You are currently OVERWRITING the last step while maintaining its reasoning direction. Please use a tone as if you are working on the previous step for the first time, and avoid wordings like \"revisit\".".to_string(),
                        StepDirection::ChangePlan => "You are currently OVERWRITING the last step while changing its reasoning direction. Please use a tone as if you are working on the previous step for the first time, and avoid wordings like \"revisit\".".to_string(),
                    }
                },
                ActualStepMode::Compact => {
                    "You are currently COMPACTING all previous steps into a more concise form. When this step is completed, it will replace all the previous steps, so please record all necessary information for future steps.".to_string()
                },
                ActualStepMode::SubmitAnswer => {
                    "You are currently REPORTING the final answer. Please leave the final answer in \\boxed{}, and then output END_STEP.".to_string()      
                }
            };
            // available tools
            // stashed subagent prompt:
            // 2. Sub-agent: The sub-agent can write and execute complex Python code for you. You're encouraged to use it for complex calculations. Make sure to provide all necessary information as the sub-agent will not have access to external context. \
            // Use the same format as this example: <tool_call>{\"name\": \"sub_agent\", \"request\": \"Find the number of prime numbers less than 1000.\"}</tool_call>.\n\
            // \n\
            let tool_prompt: String = "You can both reason in plain texts and use the following tools in this step:\n\
1. Python code executor: You're encouraged to use it for calculations to ensure correctness. Use the same format as this example: <tool_call>{\"name\": \"python\", \"code\": \"x=0\\nfor i in range(10):\\n    x += i\\nprint(x)\"}</tool_call>.\
IMPORTANT: always use print() statement to output the result, otherwise the result will not be shown.\n\
After you have output the tool call, you have to stop the generation and wait for the response.\n\
\n\
If you think a milestone has been achieved and want to mark the current step as complete, end your response with <END_STEP> and nothing else.".to_string();
            format!("Currently you are in the middle of a step.\n\
{}\n\
{}\n\
The verifier's comment is only for reference and may not be true. \
Please do not explicitly quote the verifier or try to respond to it in your reasoning.\n\
Please continue with the current step:", 
            step_mode_prompt, tool_prompt)
        }
    }
}

pub fn get_planner_prompt(
    question: &str,
    planner_status: PlannerStatus,
    history_prev_steps: &str,
    history_curr_step: &str,
) -> String {
    let planner_status_prompt = get_planner_status_prompt(planner_status);
    format!(
        "You are a planner agent that is trying to solve the following problem step by step:\n{}\n\n\
Here is the history of previous steps:\n\
{}\n\n\
{}\n\n\
{}\n",
        question, history_prev_steps, planner_status_prompt, history_curr_step
    )
}

pub fn get_verifier_prompt(
    question: &str,
    history_prev_steps: &str,
    history_curr_step: &str,
) -> String {
    format!(
        "You are a verifier agent that is trying to evaluate the reasoning steps for the following problem:\n\
{}\n\n\
Here is the history of previous steps:\n\
{}\n\n\
Your job is to generate the comment for the reasoning of the current step only. You need to provide feedbacks on the following aspects:\n\
1. Are there any calculation or logical mistakes in the current step? If so, point them out. If there are values produced, verify if they satisfy the problem requirements by plugging them in.\n\
2. Is the current reasoning direction promising for solving the problem? If not, encourage the planner to try a different direction.\n\
3. Is the length and scope of the current step appropriate? An appropriate step should have a single concrete sub-goal, neither too large nor too small. If you think the current step scope is inappropriate, make suggestions on how to plan the steps better.\n\
4. Does the current step utilize the sub_agent and python tools when necessary? If the planner attempts to do complex calculations by hand, point it out. \
Are there opportunities for the planner to use python or sub_agent for complex but pervasive problems like solving a set of linear equations? Encourage the planner to leverage these tools even when calculations appear to be manageable.\n\
\n\n\
{}\n\n\
Please start your comment and be concise:\n",
        question, history_prev_steps, history_curr_step
    )
}

fn parse_to_reasoning_and_too_calls(response: String) -> Vec<ModelOperation> {
    split_tool_call_segments(&response)
}

fn split_tool_call_segments(response: &str) -> Vec<ModelOperation> {
    let (reasonings, markdown_tool_calls) = split_markdown_python_blocks(response);
    let mut operations = Vec::new();
    let mut markdown_iter = markdown_tool_calls.into_iter();
    for reasoning in reasonings {
        operations.extend(split_tool_call_python_blocks(&reasoning));
        if let Some(markdown_call) = markdown_iter.next() {
            operations.push(ModelOperation::PlannerToolCall(markdown_call));
        }
    }
    if operations.is_empty() {
        operations.push(ModelOperation::PlannerReasoning(String::new()));
    }
    operations
}
/// Exposed for tests that need to inspect the parsed operations.
pub fn split_tool_call_segments_for_test(response: &str) -> Vec<ModelOperation> {
    split_tool_call_segments(response)
}

fn split_markdown_python_blocks(content: &str) -> (Vec<String>, Vec<String>) {
    let mut reasonings = Vec::new();
    let mut tool_calls = Vec::new();
    let mut remainder = content.trim();
    while !remainder.is_empty() {
        if let Some(start_index) = remainder.find("```python\n") {
            let reasoning_part = remainder[..start_index].trim().to_string();
            reasonings.push(reasoning_part);
            let block_start = start_index;
            let after_open_index = start_index + "```python\n".len();
            if after_open_index >= remainder.len() {
                tool_calls.push(remainder[block_start..].trim().to_string());
                break;
            }
            let after_open = &remainder[after_open_index..];
            if let Some(end_relative) = after_open.find("```") {
                let block_end = after_open_index + end_relative + "```".len();
                let block = remainder[block_start..block_end].trim().to_string();
                tool_calls.push(block);
                remainder = remainder[block_end..].trim();
            } else {
                let block = remainder[block_start..].trim().to_string();
                tool_calls.push(block);
                break;
            }
        } else {
            reasonings.push(remainder.to_string());
            remainder = "";
        }
    }
    if reasonings.is_empty() && tool_calls.is_empty() {
        reasonings.push(String::new());
    }
    (reasonings, tool_calls)
}

fn split_tool_call_python_blocks(content: &str) -> Vec<ModelOperation> {
    let mut operations = Vec::new();
    let mut remainder = content.trim().to_string();
    loop {
        if remainder.is_empty() {
            break;
        }
        if let Some(start_index) = remainder.find("<tool_call>") {
            let reasoning_part = remainder[..start_index].trim();
            if !reasoning_part.is_empty() {
                operations.push(ModelOperation::PlannerReasoning(reasoning_part.to_string()));
            }
            remainder = remainder[start_index..].trim().to_string();
            let tool_call_end_index = if let Some(end_index) = remainder[1..].find("</tool_call>") {
                assert!(
                    &remainder[end_index + 1..end_index + "</tool_call>".len() + 1]
                        == "</tool_call>"
                );
                end_index + "</tool_call>".len() + 1
            } else if let Some(next_tool_call_index) = remainder[1..].find("<tool_call>") {
                assert!(
                    &remainder
                        [next_tool_call_index + 1..next_tool_call_index + "<tool_call>".len() + 1]
                        == "<tool_call>"
                );
                next_tool_call_index + 1
            } else {
                remainder.len()
            };
            operations.push(ModelOperation::PlannerToolCall(
                remainder[..tool_call_end_index].trim().to_string(),
            ));
            remainder = remainder[tool_call_end_index..].trim().to_string();
        } else {
            let reasoning_part = remainder.trim();
            if !reasoning_part.is_empty() {
                operations.push(ModelOperation::PlannerReasoning(reasoning_part.to_string()));
            }
            break;
        }
    }
    operations
}

async fn execute_tool_call_content(tool_call_content: &str) -> String {
    // first check for valid json format
    let tool_call_json: serde_json::Value = match serde_json::from_str(tool_call_content) {
        Ok(json) => json,
        Err(_) => {
            return "<tool_response>Tool call content is not in valid json format.</tool_response>"
                .to_string();
        }
    };
    // tool call must be an object with a "name" field
    let tool_name = match tool_call_json.get("name") {
        Some(name) => match name.as_str() {
            Some("python") => "python",
            Some("sub_agent") => "sub_agent",
            _ => {
                return "<tool_response>Tool name not recognized.</tool_response>".to_string();
            }
        },
        None => {
            return "<tool_response>Tool call json must have a \"name\" field.</tool_response>"
                .to_string();
        }
    };
    match tool_name {
        "python" => {
            // for python tool, we expect a "code" field
            match tool_call_json.get("code") {
                Some(code) => match code.as_str() {
                    Some(code_str) => {
                        let python_code_result = execute_python_code(code_str.to_string()).await;
                        return format!(
                            "<tool_response>{}</tool_response>",
                            python_code_result.trim()
                        );
                    }
                    None => {
                        return "<tool_response>\"code\" field in tool call json must be a string.</tool_response>".to_string();
                    }
                },
                None => {
                    return "<tool_response>Python tool call json must have a \"code\" field.</tool_response>".to_string();
                }
            }
        }
        "sub_agent" => {
            // for sub_agent tool, we expect a "request" field
            match tool_call_json.get("request") {
                Some(request) => match request.as_str() {
                    Some(_request_str) => {
                        // for now we just return a placeholder response for the sub-agent tool, as implementing a full sub-agent is out of scope for this project
                        // return format!("<tool_response>Sub-agent received request: {}</tool_response>", request_str);
                        // TODO: implement sub-agent tool by calling rollout recursively with the request as the question
                        return format!(
                            "<tool_response>Sorry, sub-agent tool currently unavailable.</tool_response>"
                        );
                    }
                    None => {
                        return "<tool_response>\"request\" field in tool call json must be a string.</tool_response>".to_string();
                    }
                },
                None => {
                    return "<tool_response>Sub-agent tool call json must have a \"request\" field.</tool_response>".to_string();
                }
            }
        }
        _ => {
            return "<tool_response>Tool name not recognized.</tool_response>".to_string();
        }
    }
}

// there are many permutations

// we want same trajectory but whether verifier takes place

// pairs with same sampling probability, but at some point in the trajectory we ...

// sample many trajectories with the same verifier probability. For each trajectory, if it succeeds, we uniformly pick one step with verifier and remove the verifier comment
// if it fails, we uniformly pick one step without verifier and add verifier comment to it and continue

// the submit answer step should not have a verifier comment.

pub async fn rollout(
    question_id: usize,
    question: String,
    client: Client,
    model_name: &str,
    verifier_probability: f32,
    rng: &mut impl rand::Rng,
) -> Session {
    // create a state machine
    let mut session = Session::new();
    let mut safe_counter = 0; // to prevent infinite loop in case of bugs
    loop {
        let mut session_should_end = false;
        safe_counter += 1;
        if safe_counter > 20 {
            session.session_state.final_answer = Some("The model does not manage to provide a final answer within allowed number of turns.".to_string());
            session_should_end = true;
        }
        let new_operations: Vec<ModelOperation> = match &session.session_state.session_status {
            SessionStatus::PlannerTurn => {
                let history_prev_steps = session.session_state.to_history_prev_steps(true);
                let history_curr_step = session.session_state.to_history_curr_step(true);
                let planner_status = session.session_state.planner_status;
                let prompt = get_planner_prompt(
                    &question,
                    planner_status,
                    &history_prev_steps,
                    &history_curr_step,
                );
                let response = call_llm(client.clone(), prompt, model_name).await;
                match planner_status {
                    PlannerStatus::PlannerChoosingMode => {
                        let chosen_mode: ActualStepMode = match response.trim() {
                            "PROCEED" => ActualStepMode::Append(StepDirection::Proceed),
                            "CHANGE_PLAN" => ActualStepMode::Append(StepDirection::ChangePlan),
                            "OVERWRITE_LAST_STEP_AND_PROCEED" => {
                                ActualStepMode::OverwriteLastStep(StepDirection::Proceed)
                            }
                            "OVERWRITE_LAST_STEP_AND_CHANGE_PLAN" => {
                                ActualStepMode::OverwriteLastStep(StepDirection::ChangePlan)
                            }
                            "COMPACT" => ActualStepMode::Compact,
                            "SUBMIT_ANSWER" => ActualStepMode::SubmitAnswer,
                            _ => panic!("Invalid response from planner: {}", response),
                        };
                        vec![ModelOperation::PlannerChooseMode(chosen_mode)]
                    }
                    PlannerStatus::PlannerChosen(_step_mode) => {
                        let reasoning_and_tool_calls =
                            parse_to_reasoning_and_too_calls(response.clone());
                        let mut operations = reasoning_and_tool_calls;
                        let mut tool_response_operations: Vec<ModelOperation> = vec![];
                        for operation in &operations {
                            if let ModelOperation::PlannerToolCall(tool_call) = operation {
                                let tool_response = {
                                    let trimmed_tool_call = tool_call.trim_start();
                                    if trimmed_tool_call.starts_with("```python") {
                                        let fence_end_index =
                                            trimmed_tool_call.rfind("```").expect(
                                                "Markdown python tool call missing closing fence",
                                            );
                                        let code_start = trimmed_tool_call
                                            .find("\n")
                                            .map(|idx| idx + 1)
                                            .unwrap_or("```python".len());
                                        assert!(
                                            fence_end_index >= code_start,
                                            "Invalid markdown python tool call format"
                                        );
                                        let code = &trimmed_tool_call[code_start..fence_end_index];
                                        let python_code_result =
                                            execute_python_code(code.to_string()).await;
                                        format!(
                                            "<tool_response>{}</tool_response>",
                                            python_code_result.trim()
                                        )
                                    } else {
                                        if !tool_call.starts_with("<tool_call>") {
                                            panic!(
                                                "Tool call not properly formatted: {}",
                                                tool_call
                                            );
                                        }
                                        let tool_call_content_end_index = if let Some(end_index) =
                                            tool_call.find("</tool_call>")
                                        {
                                            end_index
                                        } else {
                                            // use the end index of the string
                                            tool_call.len()
                                        };
                                        let tool_call_content = &tool_call
                                            ["<tool_call>".len()..tool_call_content_end_index];
                                        execute_tool_call_content(tool_call_content).await
                                    }
                                };
                                let tool_response_operation =
                                    ModelOperation::ToolCallResponse(tool_response);
                                tool_response_operations.push(tool_response_operation);
                            }
                        }
                        operations.extend(tool_response_operations);
                        if response.contains("END_STEP") {
                            let end_step_operation = ModelOperation::PlannerEndStep;
                            operations.push(end_step_operation);
                        }
                        operations
                    }
                }
            }
            SessionStatus::VerifierTurn => {
                let mut verifier_comment = None;
                if rng.random::<f32>() <= verifier_probability
                    && !matches!(
                        session.session_state.planner_status,
                        PlannerStatus::PlannerChosen(ActualStepMode::SubmitAnswer)
                    )
                {
                    let history_prev_steps = session.session_state.to_history_prev_steps(false);
                    let history_curr_step = session.session_state.to_history_curr_step(false);
                    let prompt =
                        get_verifier_prompt(&question, &history_prev_steps, &history_curr_step);
                    let response = call_llm(client.clone(), prompt, model_name).await;
                    verifier_comment = Some(response.trim().to_string());
                }
                vec![ModelOperation::VerifierComment(verifier_comment)]
            }
        };

        for operation in new_operations {
            // println!("{}", operation.to_pretty_string());
            if session.update(operation) {
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
    // println!(
    //     "Total display rounds: {}",
    //     session.session_state.total_display_rounds()
    // );
    // println!(
    //     "Total actual rounds: {}",
    //     session.session_log.total_actual_rounds()
    // );
    // println!(
    //     "Final answer: {}",
    //     session
    //         .session_state
    //         .final_answer
    //         .as_deref()
    //         .unwrap_or("None")
    // );
    session
}
