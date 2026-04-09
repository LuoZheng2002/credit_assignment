use core::panic;

use rand::RngExt;
use reqwest::Client;

use crate::{
    apply_qwen_chat_template::apply_qwen_chat_template,
    call_llm::{call_llm_chat_completions, call_qwen_raw_completions},
    execute_python_code::execute_python_code,
    multi_agent::session::{
        ActualStepMode, ModelOperation, PlannerStatus, Session, SessionState, SessionStatus,
        StepDirection,
    },
};

#[derive(Debug, Clone, Copy)]
pub enum PlannerState {
    BeginStep,
    MidStep,
}

fn get_step_mode_prompt(step_mode: ActualStepMode, is_planner: bool) -> String {
    let subject = if is_planner {
        "You are"
    } else {
        "The planner is"
    };
    match step_mode {
        ActualStepMode::Append(step_direction) => match step_direction {
            StepDirection::Proceed => format!(
                "{} willing to proceed with the current reasoning direction.",
                subject
            ),
            StepDirection::ChangePlan => format!(
                "{} attempting a different reasoning direction from the previous steps.",
                subject
            ),
        },
        ActualStepMode::OverwriteLastStep(step_direction) => match step_direction {
            StepDirection::Proceed => format!(
                "{} about to OVERWRITE the last step while maintaining its reasoning direction. Please use a tone as if you are working on the previous step for the first time, and avoid wordings like \"revisit\".",
                subject
            ),
            StepDirection::ChangePlan => format!(
                "{} about to OVERWRITE the last step while changing its reasoning direction. Please use a tone as if you are working on the previous step for the first time, and avoid wordings like \"revisit\".",
                subject
            ),
        },
        ActualStepMode::Compact => format!(
            "{} about to COMPACT all previous steps into a more concise form. When this step is completed, it will replace all the previous steps, so please record all necessary information for future steps.",
            subject
        ),
        ActualStepMode::SubmitAnswer => format!(
            "{} about to REPORT the final answer. Please leave the final answer in \\boxed{{}}, and then output <end_step>",
            subject
        ),
    }
}

fn get_planner_status_prompt(planner_status: PlannerStatus) -> String {
    match planner_status {
        PlannerStatus::PlannerChoosingMode => {
            "\
A new step is just about to begin. Your job is to determine the mode of the new step based on the history. You have the following choices:\n\
1. SUBMIT_ANSWER: If in the previous steps, you have already found the answer, choose this option to begin the submit process.\n\
2. PROCEED: You are confident about the current reasoning direction and want to proceed with it.\n\
3. CHANGE_PLAN: You find the previous steps not leading to a good direction, and want to change the plan and try a different reasoning direction.\n\
4. OVERWRITE_LAST_STEP_AND_PROCEED: You find the last step problematic, and want to rewrite it while maintaining the current reasoning direction.\n\
5. OVERWRITE_LAST_STEP_AND_CHANGE_PLAN: You find the last step problematic, and want to rewrite it and also change the reasoning direction.\n\
6. COMPACT: You find the context length too long, and want to compact the previous steps into a more concise form, while maintaining the current reasoning direction.\n\
\n\
Please output exactly one of PROCEED, CHANGE_PLAN, OVERWRITE_LAST_STEP_AND_PROCEED, OVERWRITE_LAST_STEP_AND_CHANGE_PLAN, COMPACT, SUBMIT_ANSWER and nothing else.".to_string()
        }
        PlannerStatus::PlannerChosen(step_mode) => {
            let step_mode_prompt = get_step_mode_prompt(step_mode, true);
            let tool_prompt: String = "\
You can both reason in plain texts and use the following tools in this step:\n\
1. Python code executor: You're encouraged to use it for calculations to ensure correctness. You can invoke python code by outputting a markdown Python code block.\n\
IMPORTANT: always use Python's print statement to output the result, otherwise the result will not be shown.\n\
IMPORTANT: after calling any tool, immediately output a <tool_wait> to obtain the tool's response.\n\
\n\
If you think a milestone has been achieved and want to mark the current step as complete, end your response with <end_step> and nothing else.".to_string();
            format!("\
The verifier's comment is only for reference and may not be true. \
Please do not explicitly quote the verifier or try to respond to it in your reasoning.\n\
{}\n\
\n\
{}\n\
Please plan your new step to solve a concrete sub-goal. Avoid solving multiple sub-goals in a single step.\n\
Begin your new step:",
            tool_prompt, step_mode_prompt)
        }
    }
}

pub fn get_planner_prompt_before_assistant(
    question: &str,
    planner_status: PlannerStatus,
    history_prev_steps: &str,
) -> String {
    let planner_status_prompt = get_planner_status_prompt(planner_status);
    format!(
        "You are a planner agent that is trying to solve the following problem step by step:\n\
<PROBLEM_BEGIN>\n\
{}\n\
<PROBLEM_END>\n\n\
Here is the history of previous steps you have done:\n\
<HISTORY_BEGIN>\n\
{}\n\
<HISTORY_END>\n\n\
{}",
        question, history_prev_steps, planner_status_prompt
    )
}

pub fn get_planner_prompt_after_assistant(session_state: &SessionState) -> String {
    session_state.current_step_content_raw.clone()
}

pub fn get_verifier_prompt_before_assistant(
    question: &str,
    history_prev_steps: &str,
    session_state: &SessionState,
) -> String {
    let PlannerStatus::PlannerChosen(step_mode) = session_state.planner_status else {
        panic!("Verifier should only be called when planner has chosen the step mode");
    };
    let step_mode_prompt = get_step_mode_prompt(step_mode, false);
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
{}\n\
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
        question, history_prev_steps, step_mode_prompt, current_step_content
    )
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

// pub fn parse_to_reasoning_and_tool_calls(response: String) -> Vec<ModelOperation> {
//     split_tool_call_segments(&response)
// }

// fn split_tool_call_segments(response: &str) -> Vec<ModelOperation> {
//     let (reasonings, markdown_tool_calls) = split_markdown_python_blocks(response);
//     let mut operations = Vec::new();
//     let mut markdown_iter = markdown_tool_calls.into_iter();
//     for reasoning in reasonings {
//         operations.extend(split_tool_call_python_blocks(&reasoning));
//         if let Some(markdown_call) = markdown_iter.next() {
//             operations.push(ModelOperation::PlannerToolCall(markdown_call));
//         }
//     }
//     if operations.is_empty() {
//         operations.push(ModelOperation::PlannerReasoning(String::new()));
//     }
//     operations
// }
// /// Exposed for tests that need to inspect the parsed operations.
// pub fn split_tool_call_segments_for_test(response: &str) -> Vec<ModelOperation> {
//     split_tool_call_segments(response)
// }

// fn split_markdown_python_blocks(content: &str) -> (Vec<String>, Vec<String>) {
//     let mut reasonings = Vec::new();
//     let mut tool_calls = Vec::new();
//     let mut remainder = content.trim();
//     while !remainder.is_empty() {
//         if let Some(start_index) = remainder.find("```python\n") {
//             let reasoning_part = remainder[..start_index].trim().to_string();
//             reasonings.push(reasoning_part);
//             let block_start = start_index;
//             let after_open_index = start_index + "```python\n".len();
//             if after_open_index >= remainder.len() {
//                 tool_calls.push(remainder[block_start..].trim().to_string());
//                 break;
//             }
//             let after_open = &remainder[after_open_index..];
//             if let Some(end_relative) = after_open.find("```") {
//                 let block_end = after_open_index + end_relative + "```".len();
//                 let block = remainder[block_start..block_end].trim().to_string();
//                 tool_calls.push(block);
//                 remainder = remainder[block_end..].trim();
//             } else {
//                 let block = remainder[block_start..].trim().to_string();
//                 tool_calls.push(block);
//                 break;
//             }
//         } else {
//             reasonings.push(remainder.to_string());
//             remainder = "";
//         }
//     }
//     if reasonings.is_empty() && tool_calls.is_empty() {
//         reasonings.push(String::new());
//     }
//     (reasonings, tool_calls)
// }

// fn split_tool_call_python_blocks(content: &str) -> Vec<ModelOperation> {
//     let mut operations = Vec::new();
//     let mut remainder = content.trim().to_string();
//     loop {
//         if remainder.is_empty() {
//             break;
//         }
//         if let Some(start_index) = remainder.find("<tool_call>") {
//             let reasoning_part = remainder[..start_index].trim();
//             if !reasoning_part.is_empty() {
//                 operations.push(ModelOperation::PlannerReasoning(reasoning_part.to_string()));
//             }
//             remainder = remainder[start_index..].trim().to_string();
//             let tool_call_end_index = if let Some(end_index) = remainder[1..].find("</tool_call>") {
//                 assert!(
//                     &remainder[end_index + 1..end_index + "</tool_call>".len() + 1]
//                         == "</tool_call>"
//                 );
//                 end_index + "</tool_call>".len() + 1
//             } else if let Some(next_tool_call_index) = remainder[1..].find("<tool_call>") {
//                 assert!(
//                     &remainder
//                         [next_tool_call_index + 1..next_tool_call_index + "<tool_call>".len() + 1]
//                         == "<tool_call>"
//                 );
//                 next_tool_call_index + 1
//             } else {
//                 remainder.len()
//             };
//             operations.push(ModelOperation::PlannerToolCall(
//                 remainder[..tool_call_end_index].trim().to_string(),
//             ));
//             remainder = remainder[tool_call_end_index..].trim().to_string();
//         } else {
//             let reasoning_part = remainder.trim();
//             if !reasoning_part.is_empty() {
//                 operations.push(ModelOperation::PlannerReasoning(reasoning_part.to_string()));
//             }
//             break;
//         }
//     }
//     operations
// }

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
        if safe_counter > 30 {
            session.session_state.final_answer = Some("The model does not manage to provide a final answer within allowed number of turns.".to_string());
            session_should_end = true;
        }
        let new_operations: Vec<ModelOperation> = match &session.session_state.session_status {
            SessionStatus::PlannerTurn => {
                let history_prev_steps = session.session_state.to_history_prev_steps(true);
                let planner_status = session.session_state.planner_status;
                let prompt_before_assistant = get_planner_prompt_before_assistant(
                    &question,
                    planner_status,
                    &history_prev_steps,
                );
                let prompt_after_assistant =
                    get_planner_prompt_after_assistant(&session.session_state);
                let response = if model_name.to_lowercase().contains("qwen") {
                    let mut planner_chat_template_prompt =
                        apply_qwen_chat_template(&prompt_before_assistant);
                    planner_chat_template_prompt += &prompt_after_assistant;
                    call_qwen_raw_completions(
                        client.clone(),
                        planner_chat_template_prompt,
                        model_name,
                    )
                    .await
                } else {
                    let full_prompt = format!(
                        "{}\nAssistant: {}",
                        prompt_before_assistant, prompt_after_assistant
                    );
                    call_llm_chat_completions(client.clone(), full_prompt, model_name).await
                };
                session.add_model_raw_output(response.clone());
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
                        // let reasoning_and_tool_calls =
                        //     parse_to_reasoning_and_tool_calls(response.clone());
                        // let mut operations = reasoning_and_tool_calls;
                        // let mut tool_response_operations: Vec<ModelOperation> = vec![];
                        // for operation in &operations {
                        //     if let ModelOperation::PlannerToolCall(tool_call) = operation {
                        //         let tool_response = execute_planner_tool_call(tool_call).await;
                        //         let tool_response_operation =
                        //             ModelOperation::ToolCallResponse(tool_response);
                        //         tool_response_operations.push(tool_response_operation);
                        //     }
                        // }
                        // operations.extend(tool_response_operations);
                        // if response.contains("<end_step>") {
                        //     let end_step_operation = ModelOperation::PlannerEndStep;
                        //     operations.push(end_step_operation);
                        // }
                        let (reasoning, tool_call) =
                            split_reasoning_and_tool_call(response, model_name);
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
                    let prompt_before_assistant = get_verifier_prompt_before_assistant(
                        &question,
                        &history_prev_steps,
                        &session.session_state,
                    );
                    let response = if model_name.to_lowercase().contains("qwen") {
                        let verifier_chat_template_prompt =
                            apply_qwen_chat_template(&prompt_before_assistant);
                        call_qwen_raw_completions(
                            client.clone(),
                            verifier_chat_template_prompt,
                            model_name,
                        )
                        .await
                    } else {
                        call_llm_chat_completions(
                            client.clone(),
                            prompt_before_assistant,
                            model_name,
                        )
                        .await
                    };
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
