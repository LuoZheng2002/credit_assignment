use rand::RngExt;
use reqwest::Client;

use crate::{
    agent::{
        context_length_exceeded::is_context_length_exceeded_response,
        tool_call_parser::{MarkdownPythonParser, ToolCallParser},
        trajectory_action_types::{NextStepDecision, StepQuality, VerifierComment},
        trajectory_state::TrajectoryState,
        trajectory_status::TrajectoryStatus,
    },
    call_llm::call_llm_with_prefix,
    direct_answer::generate_raw_answers::LlmModel,
    status_prompts::universal_prompt::get_prompt_according_to_session_status,
};

// (Option<String>, Option<String>) means (reasoning, tool_call)
pub fn split_reasoning_and_tool_call(
    response: String,
) -> (Option<String>, Option<String>, bool) {
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
        return (Some(response), None, false);
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
    let mut tool_wait_violation = false;
    // if after the end position there is immediately a </tool_wait> tag, we also include it in the tool call
    if end_position < response.len() && response[end_position..].trim().starts_with("</tool_wait>")
    {
        let suffix = response[end_position..].trim_start();
        let suffix_after_tag = suffix
            .strip_prefix("</tool_wait>")
            .expect("suffix should start with </tool_wait>");
        if !suffix_after_tag.trim().is_empty() {
            println!(
                "Warning: model outputs non-empty trailing content after </tool_wait>: {}",
                suffix_after_tag.trim()
            );
            tool_wait_violation = true;
        }
        tool_call.push_str("</tool_wait>");
    } else {
        tool_wait_violation = true;

        println!("Warning: tool call does not end with </tool_wait> tag.");

        tool_call.push_str("</tool_wait>"); // if there is no </tool_wait> tag, we also add it and trim all the content after the tool call
    }
    let reasoning = if !response[..start_position].trim().is_empty() {
        Some(response[..start_position].to_string()) // do not trim the reasoning part, as leading/trailing spaces may be useful for formatting
    } else {
        None
    };
    (reasoning, Some(tool_call), tool_wait_violation)
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

pub fn parse_compactor_response(response: String) -> (String, Option<StepQuality>) {
    #[derive(serde::Deserialize)]
    struct ProperlyEndedStepQualityFields {
        tool: bool,
        complete: bool,
        focused: bool,
    }

    if let Some(json_block) = extract_content_in_json_markdown_code_block(&response) {
        if let Ok(step_quality_fields) =
            serde_json::from_str::<ProperlyEndedStepQualityFields>(json_block.trim())
        {
            let json_fence_start = response
                .find("```json")
                .expect("The json code fence start must exist if extraction succeeded");
            let summary = response[..json_fence_start].trim_end().to_string();
            return (
                summary,
                Some(StepQuality::ProperlyEnded {
                    tool: step_quality_fields.tool,
                    complete: step_quality_fields.complete,
                    focused: step_quality_fields.focused,
                }),
            );
        }
    }

    for (start_idx, _) in response.match_indices('{').rev() {
        let candidate = response[start_idx..].trim();
        if let Ok(step_quality_fields) =
            serde_json::from_str::<ProperlyEndedStepQualityFields>(candidate)
        {
            let summary = response[..start_idx].trim_end().to_string();
            return (
                summary,
                Some(StepQuality::ProperlyEnded {
                    tool: step_quality_fields.tool,
                    complete: step_quality_fields.complete,
                    focused: step_quality_fields.focused,
                }),
            );
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

pub fn parse_verifier_comment_response(response: String) -> VerifierComment {
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

pub fn get_protocol_value<'a>(response: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{}:", key);
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.to_lowercase().starts_with(&prefix) {
            return Some(trimmed[prefix.len()..].trim());
        }
    }
    None
}

pub fn parse_next_step_choice(choice: &str) -> Option<&'static str> {
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

pub enum ChosenModeDecision {
    ContextLengthExceeded,
    Chosen(NextStepDecision),
}

pub async fn determine_chosen_mode(
    session_state: &TrajectoryState<'_>,
    client: Client,
    model: LlmModel,
    take_over_mode_decision: bool,
    rng: &mut impl rand::Rng,
) -> ChosenModeDecision {
    let TrajectoryStatus::PlannerChoosingMode { verifier_comment } = &session_state.status else {
        panic!("determine_chosen_mode should only be called in PlannerChoosingMode status");
    };
    if take_over_mode_decision {
        let chosen_mode = match verifier_comment {
            None => NextStepDecision::Continue,
            Some(comment) => {
                if comment.change_plan {
                    if session_state.can_change_plan() && rng.random::<f32>() < 0.5 {
                        NextStepDecision::ChangePlan(
                            "Please refer to the verifier's comment.".to_string(),
                        )
                    } else {
                        println!("[Warning] Change plan is capped or not chosen by RNG.");
                        NextStepDecision::Continue
                    }
                } else if comment.overwrite {
                    if session_state.can_overwrite_step() && rng.random::<f32>() < 0.5 {
                        NextStepDecision::OverwriteLastStep(
                            "Please refer to the verifier's comment.".to_string(),
                        )
                    } else {
                        println!("[Warning] Overwrite last step is capped or not chosen by RNG.");
                        NextStepDecision::Continue
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
