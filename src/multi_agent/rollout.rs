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
            let tool_prompt: String = "You can both reason in plain texts and use the following tools in this step:\n\
1. Python code executor: You're encouraged to use it for simple calculations to ensure correctness. Use the same format as this example: <tool_call>{\"name\": \"python\", \"code\": \"x=0\\nfor i in range(10):\\n    x += i\\nprint(x)\"}</tool_call>.\
IMPORTANT: always use print() statement to output the result, otherwise the result will not be shown.\n\
2. Sub-agent: The sub-agent can write and execute complex Python code for you. You're encouraged to use it for complex calculations. Make sure to provide all necessary information as the sub-agent will not have access to external context. \
Use the same format as this example: <tool_call>{\"name\": \"sub_agent\", \"request\": \"Find the number of prime numbers less than 1000.\"}</tool_call>.\n\
\n\
You can only invoke one tool call at a time. After you have output the tool call, you have to stop the generation and wait for the response.\n\
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

pub fn get_planner_prompt(question: &str, planner_status: PlannerStatus, history: &str) -> String {
    let planner_status_prompt = get_planner_status_prompt(planner_status);
    format!(
        "You are a planner agent that is trying to solve the following problem step by step:\n{}\n\n\
Here is the history of previous steps:\n\
{}\n\n\
{}",
        question, history, planner_status_prompt
    )
}

pub fn get_verifier_prompt(question: &str, history: &str) -> String {
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
\n\
Please start your comment and be concise:\n",
        question, history
    )
}

fn parse_to_reasoning_and_too_calls(response: &str) -> Vec<ModelOperation> {
    let mut operations: Vec<ModelOperation> = vec![];
    loop {
        // extract non-tool-call reasoning
        let tool_call_start_index = response.find("<tool_call>");
        let Some(start_index) = tool_call_start_index else {
            let reasoning_part = response.trim();
            if !reasoning_part.is_empty() {
                operations.push(ModelOperation::PlannerReasoning(reasoning_part.to_string()));
            }
            break;
        };
        let reasoning_part = response[..start_index].trim();
        if !reasoning_part.is_empty() {
            operations.push(ModelOperation::PlannerReasoning(reasoning_part.to_string()));
        }
        // extract tool call
        let tool_call_end_index = response.find("</tool_call>").expect(
            format!(
                "Tool call start tag found without end tag, response: {}",
                response
            )
            .as_str(),
        );
        let tool_call_part =
            response[start_index..tool_call_end_index + "</tool_call>".len()].trim();
        operations.push(ModelOperation::PlannerToolCall(tool_call_part.to_string()));
        // update response by removing the processed part
        let remaining_response = response[tool_call_end_index + "</tool_call>".len()..].to_string();
        if remaining_response.trim().is_empty() {
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
                let history = session.session_state.to_history(true);
                let planner_status = session.session_state.planner_status;
                let prompt = get_planner_prompt(&question, planner_status, &history);
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
                        let reasoning_and_tool_calls = parse_to_reasoning_and_too_calls(&response);
                        let mut operations = reasoning_and_tool_calls;
                        let mut tool_response_operations: Vec<ModelOperation> = vec![];
                        for operation in &operations {
                            if let ModelOperation::PlannerToolCall(tool_call) = operation {
                                if !(tool_call.starts_with("<tool_call>")
                                    && tool_call.ends_with("</tool_call>"))
                                {
                                    panic!("Tool call not properly formatted: {}", tool_call);
                                }
                                let tool_call_content = &tool_call
                                    ["<tool_call>".len()..tool_call.len() - "</tool_call>".len()];
                                let tool_response =
                                    execute_tool_call_content(tool_call_content).await;
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
                    let history = session.session_state.to_history(false);
                    let prompt = get_verifier_prompt(&question, &history);
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
