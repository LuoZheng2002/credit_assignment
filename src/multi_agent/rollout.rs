use core::panic;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::RngExt;
use reqwest::Client;

use crate::{
    call_llm::{QWEN_CONTEXT_LENGTH_EXCEEDED_RESPONSE, call_llm_with_prefix},
    deepmath::{generate_raw_answers::Model, judge_answers::judge_answer_task},
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
            CorrectnessJudgment, MakeOrChangePlan, NextStepDecision, RolloutAction, StepQuality,
            ToolResponse, TrajectoryActionLog, TrajectoryState, TrajectoryStatus, Tree,
            TreeMasterStatus, TreeUpdateEvent,
            VerifierComment,
            CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE, IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE,
            REPETITION_ABORT_MESSAGE,
        },
        verifier_commenting::get_verifier_commenting_prompts,
    },
};

#[derive(Debug, Clone, Copy)]
pub enum PlannerState {
    BeginStep,
    MidStep,
}

fn collect_root_to_leaf_action_log(tree: &Tree, leaf_node_id: usize) -> TrajectoryActionLog {
    assert!(
        leaf_node_id < tree.nodes.len(),
        "Leaf node id must exist in tree"
    );
    let mut node_ids_from_leaf_to_root: Vec<usize> = Vec::new();
    let mut cursor = Some(leaf_node_id);
    while let Some(node_id) = cursor {
        let node = tree
            .nodes
            .get(node_id)
            .expect("Leaf-path traversal node_id must exist in tree");
        assert_eq!(
            node.node_id, node_id,
            "Node index must equal node_id during leaf-path traversal"
        );
        node_ids_from_leaf_to_root.push(node_id);
        cursor = node.parent_id;
    }
    node_ids_from_leaf_to_root.reverse();

    let mut actions: Vec<RolloutAction> = Vec::new();
    for node_id in node_ids_from_leaf_to_root {
        let node = tree
            .nodes
            .get(node_id)
            .expect("Leaf-path node_id must exist while collecting actions");
        actions.extend(node.step.action_log.iter().cloned());
    }
    TrajectoryActionLog(actions)
}

fn extract_leaf_model_answer(tree: &Tree, leaf_node_id: usize) -> String {
    let leaf_log = collect_root_to_leaf_action_log(tree, leaf_node_id);
    let leaf_state = TrajectoryState::from_session_log(tree.question.clone(), leaf_log, tree);
    leaf_state
        .final_answer
        .clone()
        .expect("Each registered leaf trajectory must have a final answer")
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
    let Some(mut start_position) = min_start_position else {
        return (Some(response), None);
    };
    // if there is <tool_call> before the start position, also include it
    if let Some(tag_position) = response[..start_position].rfind("<tool_wait>") {
        if response[tag_position..start_position].trim().is_empty() {
            start_position = tag_position;
        }
    }
    let selected_parser = selected_parser.unwrap();
    let end_position = selected_parser
        .end_position(&response, start_position)
        .unwrap_or(response.len());
    let mut tool_call = response[start_position..end_position].to_string();
    // if after the end position there is immediately a </tool_wait> tag, we also include it in the tool call
    if end_position < response.len() && response[end_position..].trim().starts_with("</tool_wait>")
    {
        tool_call.push_str("</tool_wait>");
    } else {
        if model.is_qwen() {
            println!("Warning: tool call does not end with </tool_wait> tag.");
        }
        tool_call.push_str("</tool_wait>"); // if there is no </tool_wait> tag, we also add it and trim all the content after the tool call
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

fn parse_compactor_response(response: String) -> (String, Option<StepQuality>) {
    if let Some(json_block) = extract_content_in_json_markdown_code_block(&response) {
        if let Ok(step_quality) = serde_json::from_str::<StepQuality>(json_block.trim()) {
            let json_fence_start = response
                .find("```json")
                .expect("The json code fence start must exist if extraction succeeded");
            let summary = response[..json_fence_start].trim_end().to_string();
            return (summary, Some(step_quality));
        }
    }

    for (start_idx, _) in response.match_indices('{').rev() {
        let candidate = response[start_idx..].trim();
        if let Ok(step_quality) = serde_json::from_str::<StepQuality>(candidate) {
            let summary = response[..start_idx].trim_end().to_string();
            return (summary, Some(step_quality));
        }
    }
    println!("[Warning] Failed to parse step quality from compactor response.",);
    (response, None)
}

fn parse_verifier_decision_json_manual(json_str: &str) -> Option<(bool, bool)> {
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let object = value.as_object()?;
    let mut overwrite = None;
    let mut change_plan = None;

    for (key, value) in object {
        let key_lower = key.to_lowercase();
        let Some(value_bool) = value.as_bool() else {
            continue;
        };

        if key_lower.contains("change")
            || key_lower.contains("restart")
            || key_lower.contains("plan")
        {
            change_plan = Some(value_bool);
            continue;
        }
        if key_lower.contains("overwrite")
            || key_lower.contains("rewrite")
            || key_lower.contains("last")
            || key_lower.contains("current")
            || key_lower.contains("step")
        {
            overwrite = Some(value_bool);
        }
    }

    Some((overwrite?, change_plan?))
}

fn parse_verifier_comment_response(response: String) -> VerifierComment {
    if let Some(json_block) = extract_content_in_json_markdown_code_block(&response) {
        if let Some((overwrite, change_plan)) =
            parse_verifier_decision_json_manual(json_block.trim())
        {
            let json_fence_start = response
                .find("```json")
                .expect("The json code fence start must exist if extraction succeeded");
            let comment = response[..json_fence_start].trim_end().to_string();
            return VerifierComment {
                comment,
                overwrite,
                change_plan,
            };
        }
    }

    for (start_idx, _) in response.match_indices('{').rev() {
        let candidate = response[start_idx..].trim();
        if let Some((overwrite, change_plan)) = parse_verifier_decision_json_manual(candidate) {
            let comment = response[..start_idx].trim_end().to_string();
            return VerifierComment {
                comment,
                overwrite,
                change_plan,
            };
        }
    }

    println!("[Warning] Failed to parse verifier decision JSON from verifier response.");
    VerifierComment {
        comment: response.trim().to_string(),
        overwrite: false,
        change_plan: false,
    }
}

fn get_protocol_value<'a>(response: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{}:", key);
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.to_lowercase().starts_with(&prefix) {
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
            "continue" | "proceed" => has_continue = true,
            "change" | "plan" => has_change = true,
            "overwrite" | "rewrite" | "fix" | "last" | "step" => has_overwrite = true,
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

enum ChosenModeDecision {
    ContextLengthExceeded,
    Chosen(NextStepDecision),
}

async fn determine_chosen_mode(
    session_state: &TrajectoryState<'_>,
    client: Client,
    model: Model,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
) -> ChosenModeDecision {
    if take_over_mode_decision {
        let latest_verifier_comment = session_state
            .prev_steps
            .last()
            .and_then(|step| step.current_step_verifier_comment.clone());
        let chosen_mode = match latest_verifier_comment {
            None => NextStepDecision::Continue,
            Some(comment) => {
                if comment.change_plan {
                    if rng.random::<f32>() < 0.5 {
                        NextStepDecision::Continue
                    } else {
                        NextStepDecision::ChangePlan(comment.comment)
                    }
                } else if comment.overwrite {
                    if rng.random::<f32>() < 0.5 {
                        NextStepDecision::Continue
                    } else {
                        NextStepDecision::OverwriteLastStep(comment.comment)
                    }
                } else {
                    NextStepDecision::Continue
                }
            }
        };
        return ChosenModeDecision::Chosen(chosen_mode);
    }

    let (prompt_before_assistant, prompt_after_assistant) =
        get_prompt_according_to_session_status(session_state);
    assert_eq!(
        prompt_after_assistant,
        String::new(),
        "Planner deciding next step should not have prompt after assistant"
    );
    let response = call_llm_with_prefix(
        client,
        prompt_before_assistant,
        prompt_after_assistant,
        model,
    )
    .await;
    if is_context_length_exceeded_response(&response) {
        return ChosenModeDecision::ContextLengthExceeded;
    }

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
            if session_state.can_overwrite_step() {
                let reason = get_protocol_value(trimmed_response, "reason").expect(
                    &format!(
                        "Planner choosing mode response with 'overwrite_last_step' choice does not contain 'reason: ...' line. Response: {}",
                        trimmed_response
                    ),
                );
                NextStepDecision::OverwriteLastStep(reason.to_string())
            } else {
                println!("[Warning] Overwrite last step is capped.");
                NextStepDecision::Continue
            }
        }
        "change_plan" => {
            if session_state.can_change_plan() {
                let reason = get_protocol_value(trimmed_response, "reason").expect(&format!(
                    "Planner choosing mode response with 'change_plan' choice does not contain 'reason: ...' line. Response: {}",
                    trimmed_response
                ));
                NextStepDecision::ChangePlan(reason.to_string())
            } else {
                println!("[Warning] Change plan is capped.");
                NextStepDecision::Continue
            }
        }
        _ => panic!(
            "Invalid choice field in planner choosing mode response JSON: {}",
            choice
        ),
    };
    ChosenModeDecision::Chosen(chosen_mode)
}

pub async fn execute_planner_tool_call(tool_call: &str) -> ToolResponse {
    let mut trimmed_tool_call = tool_call.trim_start().to_string();
    // trim <tool_wait>
    if trimmed_tool_call.starts_with("<tool_wait>") {
        trimmed_tool_call = trimmed_tool_call["<tool_wait>".len()..]
            .trim_start()
            .to_string();
    }
    assert!(
        trimmed_tool_call.starts_with("```python"),
        "Tool call not properly formatted: {}",
        tool_call
    );
    let Some(fence_end_index) = trimmed_tool_call.rfind("```") else {
        return ToolResponse::PythonError(
            "Tool call markdown code block not properly closed.".to_string(),
        );
    };
    let code_start = trimmed_tool_call
        .find('\n')
        .map(|idx| idx + 1)
        .unwrap_or("```python".len());
    if fence_end_index < code_start {
        return ToolResponse::PythonError(
            "Tool call markdown code block not properly formatted.".to_string(),
        );
    }
    let code = &trimmed_tool_call[code_start..fence_end_index];
    execute_python_code(code.to_string()).await
}

pub fn get_prompt_according_to_session_status(
    session_state: &TrajectoryState<'_>,
) -> (String, String) {
    match session_state.status {
        TrajectoryStatus::PlannerMakingOrChangingPlan => {
            get_planner_making_plan_prompts(session_state)
        }
        TrajectoryStatus::PlannerKeepingCurrentPlan => (String::new(), String::new()),
        TrajectoryStatus::PlannerChoosingMode => {
            get_planner_deciding_next_step_prompts(session_state)
        }
        TrajectoryStatus::PlannerWorkingOnStep => {
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
                    get_planner_step_continuing_prompts(session_state)
                }
            }
        }
        TrajectoryStatus::PlannerCompactingStep => get_planner_compacting_prompts(session_state),
        TrajectoryStatus::PlannerUpdatingPlan => get_planner_updating_plan_prompts(session_state),
        TrajectoryStatus::VerifierCommenting => get_verifier_commenting_prompts(session_state),
    }
}

pub const SUBMIT_ANSWER_HINT: &str = "\
<hint>It seems you are trying to end the step at the start of the step. \
If you have got the answer, put it in \\boxed{} before ending with <end_step>.</hint>";
pub const MAX_NUM_TRAJECTORIES: usize = 16;

// we increased the repetition times to 5, there might be code that hasn't reflected this change.
pub fn detect_repetition_five_times(response: &str) -> bool {
    let min_subsequence_length = 50; // minimum length of the repeated subsequence to avoid false positive from short common phrases

    let bytes = response.as_bytes();
    let n = bytes.len();
    if n < min_subsequence_length * 5 {
        return false;
    }

    // Rolling hash over raw bytes. We still verify by byte comparison when hashes match.
    let base: u64 = 1_000_003;
    let mut pow = vec![0_u64; n + 1];
    let mut prefix = vec![0_u64; n + 1];
    pow[0] = 1;
    for i in 0..n {
        pow[i + 1] = pow[i].wrapping_mul(base);
        prefix[i + 1] = prefix[i]
            .wrapping_mul(base)
            .wrapping_add((bytes[i] as u64) + 1);
    }

    let hash = |start: usize, len: usize| -> u64 {
        prefix[start + len].wrapping_sub(prefix[start].wrapping_mul(pow[len]))
    };

    for len in min_subsequence_length..=(n / 5) {
        for start in 0..=(n - 5 * len) {
            let h1 = hash(start, len);
            let h2 = hash(start + len, len);
            let h3 = hash(start + 2 * len, len);
            let h4 = hash(start + 3 * len, len);
            let h5 = hash(start + 4 * len, len);
            if h1 != h2 || h1 != h3 || h1 != h4 || h1 != h5 {
                continue;
            }

            let s1 = &bytes[start..start + len];
            let s2 = &bytes[start + len..start + 2 * len];
            let s3 = &bytes[start + 2 * len..start + 3 * len];
            let s4 = &bytes[start + 3 * len..start + 4 * len];
            let s5 = &bytes[start + 4 * len..start + 5 * len];
            if s1 == s2 && s1 == s3 && s1 == s4 && s1 == s5 {
                return true;
            }
        }
    }

    false
}

fn is_context_length_exceeded_response(response: &str) -> bool {
    response == QWEN_CONTEXT_LENGTH_EXCEEDED_RESPONSE
}

fn context_length_exceeded_result(question_id: usize, session_status: &str) -> Vec<TreeUpdateEvent> {
    println!(
        "[Warning] Model context length exceeded in {}, ending session.",
        session_status
    );
    vec![TreeUpdateEvent::AddAction {
        question_id,
        action: RolloutAction::ToolCallResponse(ToolResponse::Intervention(
            CONTEXT_LENGTH_EXCEEDED_ABORT_MESSAGE.to_string(),
        )),
    }]
}

async fn build_new_operations(
    session_state: &TrajectoryState<'_>,
    client: Client,
    model: Model,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
) -> Vec<TreeUpdateEvent> {
    let question_id = session_state.source_tree.question_id;
    match &session_state.status {
        TrajectoryStatus::PlannerMakingOrChangingPlan => {
            let chosen_mode = session_state
                .planner_chosen_mode
                .as_ref()
                .expect("planner_chosen_mode must be set before PlannerMakingOrChangingPlan");
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(session_state);
            let response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id, "PlannerMakingOrChangingPlan")
            } else {
                let plan_content = match chosen_mode {
                    NextStepDecision::ChangePlan(reason) => Some(MakeOrChangePlan::ChangePlan {
                        plan: response,
                        prev_failed_reason: reason.clone(),
                    }),
                    NextStepDecision::Continue | NextStepDecision::OverwriteLastStep(_) => {
                        Some(MakeOrChangePlan::MakePlan(response))
                    }
                }; // we change to not require the plan to be in a markdown code block
                vec![TreeUpdateEvent::AddAction {
                    question_id,
                    action: RolloutAction::PlannerMakeOrChangePlan(plan_content),
                }]
            }
        }
        TrajectoryStatus::PlannerKeepingCurrentPlan => vec![TreeUpdateEvent::AddAction {
                question_id,
                action: RolloutAction::PlannerMakeOrChangePlan(None),
            }],
        TrajectoryStatus::PlannerChoosingMode => {
            match determine_chosen_mode(
                session_state,
                client.clone(),
                model,
                take_over_mode_decision,
                rng,
            )
            .await
            {
                ChosenModeDecision::ContextLengthExceeded => {
                    context_length_exceeded_result(question_id, "PlannerChoosingMode")
                }
                ChosenModeDecision::Chosen(chosen_mode) => vec![TreeUpdateEvent::AddAction {
                    question_id,
                    action: RolloutAction::PlannerDecideNextStep(chosen_mode),
                }],
            }
        }
        TrajectoryStatus::PlannerWorkingOnStep => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(session_state);
            let mut response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id, "PlannerWorkingOnStep")
            } else {
                if response.trim().is_empty() {
                    response = "<end_step>".to_string(); // if the model does not output anything, we treat it as if it outputs <end_step> to prevent getting stuck
                }
                let mut hold_end_step = false;
                if &response == "<end_step>" && &session_state.current_step_content_raw == "" {
                    println!(
                        "[Warning]: model tries to end the step without providing any content for the step."
                    );
                    hold_end_step = true;
                }
                let (reasoning, tool_call) = split_reasoning_and_tool_call(response.clone(), model);
                let mut push_end_step = false;
                let mut has_terminal_intervention = false;
                let mut operations: Vec<TreeUpdateEvent> = Vec::new();
                if let Some(reasoning) = reasoning {
                    if reasoning.contains("<end_step>") {
                        push_end_step = true;
                    }
                    operations.push(TreeUpdateEvent::AddAction {
                        question_id,
                        action: RolloutAction::PlannerReasoning(reasoning),
                    });
                }
                if let Some(tool_call) = tool_call {
                    let tool_response = execute_planner_tool_call(&tool_call).await;
                    let previous_python_error = session_state.current_step_last_python_error.clone();
                    operations.push(TreeUpdateEvent::AddAction {
                        question_id,
                        action: RolloutAction::PlannerToolCall(tool_call),
                    });
                    if let ToolResponse::PythonError(current_python_error) = &tool_response {
                        if previous_python_error.is_some()
                            && Some(current_python_error.clone()) == previous_python_error
                        {
                            println!(
                                "[Warning]: Identical python tool error detected. Aborting current step."
                            );
                            operations.push(TreeUpdateEvent::AddAction {
                                question_id,
                                action: RolloutAction::ToolCallResponse(tool_response),
                            });
                            operations.push(TreeUpdateEvent::AddAction {
                                question_id,
                                action: RolloutAction::ToolCallResponse(ToolResponse::Intervention(
                                    IDENTICAL_PYTHON_ERROR_ABORT_MESSAGE.to_string(),
                                )),
                            });
                            has_terminal_intervention = true;
                        } else {
                            operations.push(TreeUpdateEvent::AddAction {
                                question_id,
                                action: RolloutAction::ToolCallResponse(tool_response),
                            });
                        }
                    } else {
                        operations.push(TreeUpdateEvent::AddAction {
                            question_id,
                            action: RolloutAction::ToolCallResponse(tool_response),
                        });
                    }
                }
                if hold_end_step {
                    operations.push(TreeUpdateEvent::AddAction {
                        question_id,
                        action: RolloutAction::ToolCallResponse(ToolResponse::Intervention(
                            SUBMIT_ANSWER_HINT.to_string(),
                        )),
                    });
                }
                let num_additional_actions_allowed =
                    session_state.num_additional_actions_allowed_in_current_step();
                if operations.len() > num_additional_actions_allowed {
                    println!(
                        "[Warning] Number of actions in the current step {} exceeds the limit {}. Only the first {} actions will be applied.",
                        operations.len(),
                        num_additional_actions_allowed,
                        num_additional_actions_allowed
                    );
                    operations.truncate(num_additional_actions_allowed);
                }
                // detect repetition
                let found_repetition_three_times = detect_repetition_five_times(&response);
                if found_repetition_three_times {
                    println!(
                        "[Warning] Detected repetition of the same response at least three times. This may indicate that the model is stuck in a loop. Response: {}",
                        response
                    );
                    operations.push(TreeUpdateEvent::AddAction {
                        question_id,
                        action: RolloutAction::ToolCallResponse(ToolResponse::Intervention(
                            REPETITION_ABORT_MESSAGE.to_string(),
                        )),
                    });
                    has_terminal_intervention = true;
                }

                let current_step_full = operations.len() == num_additional_actions_allowed;
                if (push_end_step && !hold_end_step)
                    || current_step_full
                {
                    assert!(
                        !has_terminal_intervention,
                        "PlannerEndStep should not be emitted after terminal intervention"
                    );
                    operations.push(TreeUpdateEvent::AddAction {
                        question_id,
                        action: RolloutAction::PlannerEndStep,
                    });
                }
                operations
            }
        }
        TrajectoryStatus::PlannerCompactingStep => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(session_state);
            let response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id, "PlannerCompactingStep")
            } else {
                let (summary, step_quality) = parse_compactor_response(response);
                vec![TreeUpdateEvent::AddAction {
                    question_id,
                    action: RolloutAction::PlannerCompactStep {
                        summary,
                        step_quality,
                    },
                }]
            }
        }
        TrajectoryStatus::PlannerUpdatingPlan => {
            let (prompt_before_assistant, prompt_after_assistant) =
                get_prompt_according_to_session_status(session_state);
            let response = call_llm_with_prefix(
                client.clone(),
                prompt_before_assistant,
                prompt_after_assistant,
                model,
            )
            .await;
            if is_context_length_exceeded_response(&response) {
                context_length_exceeded_result(question_id, "PlannerUpdatingPlan")
            } else {
                let updated_plan_content = response; // we change to not require the updated plan to be in a markdown code block
                vec![TreeUpdateEvent::AddAction {
                    question_id,
                    action: RolloutAction::PlannerUpdatePlan(updated_plan_content),
                }]
            }
        }
        TrajectoryStatus::VerifierCommenting => {
            let mut operations: Vec<TreeUpdateEvent> = Vec::new();
            let (node_id, parent_id) = if session_state.source_tree.current_node_id.is_none() {
                assert!(
                    session_state.source_tree.root_node_id.is_none()
                        && session_state.source_tree.nodes.is_empty()
                        && session_state.source_tree.next_node_id == 0,
                    "Tree without current node must be an uninitialized empty tree"
                );
                (0, None)
            } else {
                let parent_node_id = session_state
                    .source_tree
                    .current_node_id
                    .expect("VerifierCommenting requires current_node_id");
                let next_node_id = session_state.source_tree.next_node_id;
                (next_node_id, Some(parent_node_id))
            };
            let verifier_on = if let Some(parent_node_id) = parent_id {
                let parent_node = session_state
                    .source_tree
                    .nodes
                    .get(parent_node_id)
                    .expect("VerifierCommenting parent node must exist");
                assert_eq!(
                    parent_node.node_id, parent_node_id,
                    "Node index must equal node_id when choosing child branch"
                );
                match (
                    parent_node.verifier_on_child_id.is_some(),
                    parent_node.verifier_off_child_id.is_some(),
                ) {
                    (false, false) => Some(rng.random::<f32>() < 0.5),
                    (false, true) => Some(true),
                    (true, false) => Some(false),
                    (true, true) => panic!(
                        "VerifierCommenting parent already has both verifier_on and verifier_off children"
                    ),
                }
            } else {
                None
            };
            operations.push(TreeUpdateEvent::CreateNode {
                question_id,
                node_id,
                parent_id,
                verifier_on,
            });
            operations.push(TreeUpdateEvent::SetCurrentNode {
                question_id,
                node_id,
            });
            let mut verifier_comment = None;
            let should_run_verifier = verifier_on == Some(true);
            if !session_state.prev_steps.is_empty() && should_run_verifier {
                let (prompt_before_assistant, prompt_after_assistant) =
                    get_prompt_according_to_session_status(session_state);
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
                if is_context_length_exceeded_response(&response) {
                    operations.extend(context_length_exceeded_result(
                        question_id,
                        "VerifierCommenting",
                    ));
                    return operations;
                }
                verifier_comment = Some(parse_verifier_comment_response(response));
            }
            operations.push(TreeUpdateEvent::AddAction {
                question_id,
                action: RolloutAction::VerifierComment(verifier_comment),
            });
            operations
        }
    }
}

pub async fn produce_working_trajectory(
    tree: &mut Tree,
    reference_answer: &str,
    client: &Client,
    model: Model,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
    action_tx: &tokio::sync::mpsc::UnboundedSender<TreeUpdateEvent>,
) {
    assert_eq!(
        tree.tree_master_status,
        TreeMasterStatus::WorkingOnTrajectory,
        "produce_working_trajectory requires WorkingOnTrajectory status"
    );
    loop {
        let session_state = TrajectoryState::from_tree(tree);
        println!(
            "[rollout] question index: {}, num actions: {}, num prev steps: {}, num actual steps: {}",
            tree.question_id,
            session_state.total_actions,
            session_state.prev_steps.len(),
            session_state.total_actual_steps
        );
        if session_state.should_end_session {
            let displayed_final_answer = session_state.final_answer.as_deref().unwrap_or("None");
            println!(
                "[rollout finished] question index: {}, total actual rounds: {}, final answer: {}, correct answer: {}",
                tree.question_id,
                session_state.total_actions,
                displayed_final_answer,
                reference_answer
            );
            let leaf_node_id = tree
                .current_node_id
                .expect("WorkingOnTrajectory should always have current_node_id when ending");
            if !tree.leaf_node_ids.contains(&leaf_node_id) {
                let register_leaf_event = TreeUpdateEvent::RegisterLeaf {
                    question_id: tree.question_id,
                    node_id: leaf_node_id,
                };
                tree.apply_event(register_leaf_event.clone());
                action_tx.send(register_leaf_event).unwrap();
            }
            if !tree.leaf_node_judgments.contains_key(&leaf_node_id) {
                let model_answer = extract_leaf_model_answer(tree, leaf_node_id);
                let is_correct = judge_answer_task(
                    tree.question_id,
                    model_answer.clone(),
                    reference_answer.to_string(),
                    tree.question.clone(),
                    client.clone(),
                )
                .await;
                let judge_leaf_correctness_event = TreeUpdateEvent::JudgeLeafCorrectness {
                    question_id: tree.question_id,
                    node_id: leaf_node_id,
                    correctness_judgment: CorrectnessJudgment {
                        model_answer,
                        correct_answer: reference_answer.to_string(),
                        is_correct,
                    },
                };
                tree.apply_event(judge_leaf_correctness_event.clone());
                action_tx.send(judge_leaf_correctness_event).unwrap();
            }
            break;
        }
        let new_operations = build_new_operations(
            &session_state,
            client.clone(),
            model,
            take_over_mode_decision,
            rng,
        )
        .await;
        drop(session_state);

        for event in new_operations {
            tree.apply_event(event.clone());
            action_tx.send(event).unwrap();
        }
    }
}

pub fn determine_branching_node(
    tree: &mut Tree,
    rng: &mut impl rand::Rng,
) -> bool {
    assert_eq!(
        tree.tree_master_status,
        TreeMasterStatus::DeterminingBranchingNode,
        "determine_branching_node requires DeterminingBranchingNode status"
    );
    if tree.leaf_node_ids.len() >= MAX_NUM_TRAJECTORIES {
        return true;
    }

    let mut node_weights = vec![0.0_f64; tree.nodes.len()];
    let mut trajectory_lengths: Vec<usize> = Vec::new();
    for &trajectory_leaf_node_id in &tree.leaf_node_ids {
        let mut trajectory_node_ids_from_leaf_to_root: Vec<usize> = Vec::new();
        let mut cursor = Some(trajectory_leaf_node_id);
        while let Some(node_id) = cursor {
            trajectory_node_ids_from_leaf_to_root.push(node_id);
            let node = tree
                .nodes
                .get(node_id)
                .expect("Trajectory traversal node_id must exist");
            assert_eq!(
                node.node_id, node_id,
                "Node index must equal node_id during trajectory traversal"
            );
            cursor = node.parent_id;
        }
        trajectory_node_ids_from_leaf_to_root.reverse();
        trajectory_lengths.push(trajectory_node_ids_from_leaf_to_root.len());
        if trajectory_node_ids_from_leaf_to_root.len() < 2 {
            return true;
        }
        let per_node_weight = 1.0 / (trajectory_node_ids_from_leaf_to_root.len() - 1) as f64;
        let non_leaf_node_ids =
            &trajectory_node_ids_from_leaf_to_root[..trajectory_node_ids_from_leaf_to_root.len() - 1];
        for &node_id in non_leaf_node_ids {
            node_weights[node_id] += per_node_weight;
        }
    }
    assert_eq!(
        trajectory_lengths.len(),
        tree.leaf_node_ids.len(),
        "Each leaf trajectory should contribute one trajectory length"
    );

    let mut candidate_node_ids: Vec<usize> = Vec::new();
    let mut candidate_weights: Vec<f64> = Vec::new();
    for node in &tree.nodes {
        let weight = node_weights[node.node_id];
        if weight > 0.0 {
            candidate_node_ids.push(node.node_id);
            candidate_weights.push(weight);
        }
    }
    while !candidate_node_ids.is_empty() {
        let weighted_index = WeightedIndex::new(&candidate_weights)
            .expect("WeightedIndex construction should succeed with positive candidate weights");
        let sampled_candidate_index = weighted_index.sample(rng);
        let selected_node_id = candidate_node_ids[sampled_candidate_index];
        let selected_node = tree
            .nodes
            .get(selected_node_id)
            .expect("Selected branching node must exist");
        assert_eq!(
            selected_node.node_id, selected_node_id,
            "Node index must equal node_id for selected branching node"
        );
        let has_verifier_on_child = selected_node.verifier_on_child_id.is_some();
        let has_verifier_off_child = selected_node.verifier_off_child_id.is_some();
        if has_verifier_on_child && has_verifier_off_child {
            candidate_node_ids.swap_remove(sampled_candidate_index);
            candidate_weights.swap_remove(sampled_candidate_index);
            continue;
        }
        tree.set_current_node_by_id(selected_node_id);
        tree.tree_master_status = TreeMasterStatus::WorkingOnTrajectory;
        return false;
    }

    return true;
}

// it will output action logs and final trajectory
// it will also load existing logs
pub async fn rollout(
    question_id: usize,
    question: String,
    reference_answer: String,
    loaded_events: Vec<TreeUpdateEvent>,
    client: Client,
    model: Model,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
    action_tx: tokio::sync::mpsc::UnboundedSender<TreeUpdateEvent>,
    trajectory_tx: tokio::sync::mpsc::UnboundedSender<RolloutTrajectory>,
) {
    // create a state machine
    let mut tree = Tree::new(question_id, question.clone());
    for event in loaded_events {
        tree.apply_event(event);
    }
    loop {
        match tree.tree_master_status {
            TreeMasterStatus::WorkingOnTrajectory => {
                produce_working_trajectory(
                    &mut tree,
                    &reference_answer,
                    &client,
                    model,
                    take_over_mode_decision,
                    rng,
                    &action_tx,
                )
                .await;
                tree.tree_master_status = TreeMasterStatus::DeterminingBranchingNode;
            }
            TreeMasterStatus::DeterminingBranchingNode => {
                let should_finalize_rollout = determine_branching_node(&mut tree, rng);
                if should_finalize_rollout {
                    break;
                }
            }
        }
    }
    let final_state = TrajectoryState::from_tree(&tree);
    let step_quality_accuracy = final_state.step_quality_accuracy();
    let trajectory_tree = tree.clone();
    let rollout_trajectory = RolloutTrajectory {
        id: question_id,
        question,
        model_answer: final_state
            .final_answer
            .clone()
            .unwrap_or("No answer found".into()),
        correct_answer: reference_answer,
        step_quality_accuracy,
        trajectory: trajectory_tree,
    };
    trajectory_tx.send(rollout_trajectory).unwrap();
}
