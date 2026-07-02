use reqwest::Client;
use research_utility::progress_tui_logger::log_warning;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::{Duration, sleep};

use crate::{
    atomic_count_guard::AtomicCountGuardRef,
    direct_tool::{rollout::RolloutStats, trajectory::FinalAnswer},
    model_answer_judgment_cache::{get_cached_judgment, store_cached_judgment},
};

const DEEPSEEK_CHAT_COMPLETIONS_URL: &str = "https://api.deepseek.com/v1/chat/completions";
const OPENROUTER_CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const DEEPSEEK_V4_PRO_MODEL: &str = "deepseek-v4-pro";
const DEEPSEEK_V4_PRO_OPENROUTER_MODEL: &str = "deepseek/deepseek-v4-pro";
const USE_OPENROUTER_API: bool = true;
const JUDGE_TOTAL_ATTEMPTS: usize = 10;

/// Extract the content inside the **last** `\boxed{...}` in the response.
/// Returns the trimmed inner text, or `None` if no boxed content is found.
fn extract_boxed_verdict(response: &str) -> Option<String> {
    let marker = "\\boxed{";
    let start = response.rfind(marker)?;
    let content_start = start + marker.len();
    let remaining = &response[content_start..];
    let mut depth: u32 = 1;
    for (i, c) in remaining.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(remaining[..i].trim().to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessJudgment {
    pub model_answer: FinalAnswer,
    pub correct_answer: String,
    pub is_correct: bool,
}

async fn judge_answer_task(
    model_answer: String,
    correct_answer: String,
    question: String,
    client: Client,
) -> bool {
    let url = if USE_OPENROUTER_API {
        OPENROUTER_CHAT_COMPLETIONS_URL
    } else {
        DEEPSEEK_CHAT_COMPLETIONS_URL
    };
    judge_answer_task_with_url(model_answer, correct_answer, question, client, url).await
}

async fn judge_answer_task_with_url(
    model_answer: String,
    correct_answer: String,
    question: String,
    client: Client,
    url: &str,
) -> bool {
    let base_prompt = format!(
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
Do not attempt to solve the problem yourself, only judge whether the given answer and the reference answer is equivalent. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
The question is: \"{}\". \
The model's answer is: \"{}\", and the correct answer is: \"{}\".",
        question, model_answer, correct_answer
    );
    let thinking_prompt = format!(
        "{base_prompt} Think step by step about whether the model's answer matches the reference answer. Put your final answer in \\boxed{{correct}} or \\boxed{{incorrect}}."
    );

    let mut last_error: Option<String> = None;

    for attempt in 0..JUDGE_TOTAL_ATTEMPTS {
        let prompt = &thinking_prompt;
        let temperature = if attempt == 0 { 0.0 } else { 1.0 };
        let thinking_enabled = false;

        let fetch_result =
            fetch_judge_evaluation_with_url(&client, prompt, url, temperature, thinking_enabled)
                .await;
        match fetch_result {
            Ok((evaluation, _reasoning)) => match extract_boxed_verdict(&evaluation) {
                Some(verdict) => {
                    let verdict_lower = verdict.to_lowercase();
                    if verdict_lower.contains("incorrect") {
                        return false;
                    }
                    if verdict_lower.contains("correct") {
                        return true;
                    }
                    last_error = Some(format!(
                        "Verdict in \\boxed{{}} was neither 'correct' nor 'incorrect': {}",
                        verdict
                    ));
                }
                None => {
                    last_error = Some(format!(
                        "No \\boxed{{}} found in evaluation response: {}",
                        evaluation
                    ));
                }
            },
            Err(error) => {
                last_error = Some(error.to_string());
            }
        }
        log_warning(format!(
            "Judger returned invalid response, attempt {}/{}. Last error: {}",
            attempt + 1,
            JUDGE_TOTAL_ATTEMPTS,
            last_error
                .as_deref()
                .unwrap_or("none (response did not include 'correct' or 'incorrect')")
        ));

        if attempt + 1 < JUDGE_TOTAL_ATTEMPTS {
            sleep(Duration::from_secs(1)).await;
        }
    }

    panic!(
        "Failed to judge answer after {} attempts: {}",
        JUDGE_TOTAL_ATTEMPTS,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    );
}

async fn fetch_judge_evaluation_with_url(
    client: &Client,
    prompt: &str,
    url: &str,
    temperature: f64,
    thinking_enabled: bool,
) -> Result<(String, Option<String>), String> {
    let model_name = if USE_OPENROUTER_API {
        DEEPSEEK_V4_PRO_OPENROUTER_MODEL
    } else {
        DEEPSEEK_V4_PRO_MODEL
    };
    let api_key_env = if USE_OPENROUTER_API {
        "OPENROUTER_API_KEY"
    } else {
        "DEEPSEEK_API_KEY"
    };

    let api_key = std::env::var(api_key_env)
        .map_err(|_| format!("{api_key_env} environment variable not set"))?;
    let mut body_map = serde_json::json!({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 4096,
        "temperature": temperature,
    });
    if USE_OPENROUTER_API {
        body_map["reasoning"] = serde_json::json!({
            "effort": if thinking_enabled { "high" } else { "none" }
        });
    } else {
        body_map["thinking"] = serde_json::json!({
            "type": if thinking_enabled { "enabled" } else { "disabled" }
        });
    }

    let mut request_builder = client.post(url).json(&body_map);
    request_builder = request_builder.bearer_auth(api_key);
    request_builder = request_builder
        .header("HTTP-Referer", "https://github.com/luoz/credit_assignment")
        .header("X-Title", "credit_assignment");

    let response = request_builder
        .send()
        .await
        .map_err(|err| format!("Failed to send judge request: {err}"))?;
    let response_bytes = response
        .bytes()
        .await
        .map_err(|err| format!("Failed to read judge response body: {err}"))?;
    let response_json = serde_json::from_slice::<Value>(&response_bytes).map_err(|_| {
        format!(
            "Failed to parse judge response as JSON. Response text: {:?}",
            String::from_utf8_lossy(&response_bytes)
        )
    })?;

    if let Some(error_message) = response_json["error"]["message"].as_str() {
        return Err(error_message.to_string());
    }

    // OpenRouter responds with "reasoning", DeepSeek official responds with "reasoning_content".
    let reasoning_content = response_json["choices"][0]["message"]
        .as_object()
        .and_then(|msg| {
            msg.get("reasoning_content")
                .or_else(|| msg.get("reasoning"))
                .and_then(|v| v.as_str())
        })
        .and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

    let content = &response_json["choices"][0]["message"]["content"];
    if let Some(evaluation) = content.as_str() {
        return Ok((evaluation.to_string(), reasoning_content));
    }
    if let Some(parts) = content.as_array() {
        let merged = parts
            .iter()
            .filter_map(|entry| entry["text"].as_str())
            .collect::<String>();
        if !merged.is_empty() {
            return Ok((merged, reasoning_content));
        }
    }

    Err(format!("Judge response is invalid: {response_json:?}"))
}

pub async fn judge_final_answer(
    final_answer: &FinalAnswer,
    correct_answer: &str,
    question: &str,
    client: Client,
) -> CorrectnessJudgment {
    let rollout_stats = RolloutStats::global();
    let is_correct = match final_answer {
        FinalAnswer::ModelProvided(model_answer) => {
            let _num_judge_waiting_workers_guard = AtomicCountGuardRef::new(
                &rollout_stats.judge_waiting_workers,
                "judge_waiting_workers".to_string(),
            );

            let cache_lookup = get_cached_judgment(question, model_answer.clone());
            match cache_lookup {
                Ok(Some(is_correct)) => {
                    rollout_stats.record_model_answer_judgment_cache_read_attempt(true);
                    is_correct
                }
                Ok(None) => {
                    rollout_stats.record_model_answer_judgment_cache_read_attempt(false);
                    let is_correct = judge_answer_task(
                        model_answer.clone(),
                        correct_answer.to_string(),
                        question.to_string(),
                        client,
                    )
                    .await;
                    if let Err(error) =
                        store_cached_judgment(question, model_answer.clone(), is_correct)
                    {
                        log_warning(format!(
                            "Failed to store model answer judgment cache entry for question {:?}: {}",
                            question, error
                        ));
                    }
                    is_correct
                }
                Err(error) => {
                    rollout_stats.record_model_answer_judgment_cache_read_attempt(false);
                    log_warning(format!(
                        "Failed to read model answer judgment cache entry for question {:?}: {}",
                        question, error
                    ));
                    let is_correct = judge_answer_task(
                        model_answer.clone(),
                        correct_answer.to_string(),
                        question.to_string(),
                        client,
                    )
                    .await;
                    if let Err(error) =
                        store_cached_judgment(question, model_answer.clone(), is_correct)
                    {
                        log_warning(format!(
                            "Failed to store model answer judgment cache entry for question {:?}: {}",
                            question, error
                        ));
                    }
                    is_correct
                }
            }
        }
        FinalAnswer::Failure(_error_message) => false,
    };
    CorrectnessJudgment {
        model_answer: final_answer.clone(),
        correct_answer: correct_answer.to_string(),
        is_correct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judge_api_url() -> &'static str {
        if USE_OPENROUTER_API {
            OPENROUTER_CHAT_COMPLETIONS_URL
        } else {
            DEEPSEEK_CHAT_COMPLETIONS_URL
        }
    }

    /// Helper: call `fetch_judge_evaluation_with_url` against the real
    /// API, panicking on failure (so test failure messages are clear).
    async fn call_api(
        client: &Client,
        prompt: &str,
        temperature: f64,
        thinking_enabled: bool,
    ) -> (String, Option<String>) {
        fetch_judge_evaluation_with_url(
            client,
            prompt,
            judge_api_url(),
            temperature,
            thinking_enabled,
        )
        .await
        .expect("API call should succeed")
    }

    /// Helper: run `judge_answer_task_with_url` against the real API.
    async fn call_judge(
        client: &Client,
        model_answer: &str,
        correct_answer: &str,
        question: &str,
    ) -> bool {
        judge_answer_task_with_url(
            model_answer.to_string(),
            correct_answer.to_string(),
            question.to_string(),
            client.clone(),
            judge_api_url(),
        )
        .await
    }

    /// Thinking mode (temperature=0, thinking enabled) should return a valid
    /// answer and non-empty reasoning content.
    #[tokio::test]
    async fn test_thinking_mode() {
        let _ = dotenvy::dotenv();
        let client = Client::new();
        let (content, reasoning) = call_api(
            &client,
            "A student says the answer is '4'. The answer key says '4'. Think step by step and put your verdict in \\boxed{correct} or \\boxed{incorrect}.",
            0.0,
            true,
        )
        .await;
        let lower = content.trim().to_lowercase();
        assert!(
            lower.contains("correct") || lower.contains("incorrect"),
            "thinking mode should return a valid verdict, got: {content:?}"
        );
        assert!(
            reasoning.is_some(),
            "reasoning should be present when thinking is enabled"
        );
    }

    /// The full judge pipeline should return true when the model answer matches
    /// the correct answer.
    #[tokio::test]
    async fn test_judge_correct_answer() {
        let _ = dotenvy::dotenv();
        let client = Client::new();
        let result = call_judge(&client, "4", "4", "What is 2+2?").await;
        assert!(result, "judge should return true for matching answers");
    }

    /// The full judge pipeline should return false when the model answer does
    /// not match the correct answer.
    #[tokio::test]
    async fn test_judge_incorrect_answer() {
        let _ = dotenvy::dotenv();
        let client = Client::new();
        let result = call_judge(&client, "5", "4", "What is 2+2?").await;
        assert!(!result, "judge should return false for mismatched answers");
    }

    /// Two identical calls at temperature 0 with thinking enabled should
    /// converge on the same verdict.  The model may still produce
    /// byte-for-byte different output (no `seed` parameter), but the final
    /// conclusion should be consistent.  We also verify that reasoning content
    /// is present.
    #[tokio::test]
    async fn test_thinking_consistent_verdict() {
        let _ = dotenvy::dotenv();
        let client = Client::new();
        let prompt = concat!(
            "Verify step by step: does '42' equal '42'? ",
            "Put your final verdict in \\boxed{correct} or \\boxed{incorrect}."
        );

        let (content_a, reasoning_a) = call_api(&client, prompt, 0.0, true).await;
        let (content_b, reasoning_b) = call_api(&client, prompt, 0.0, true).await;

        let lower_a = content_a.trim().to_lowercase();
        let lower_b = content_b.trim().to_lowercase();
        assert!(
            lower_a.contains("correct") || lower_a.contains("incorrect"),
            "call A verdict missing, got: {content_a:?}"
        );
        assert!(
            lower_b.contains("correct") || lower_b.contains("incorrect"),
            "call B verdict missing, got: {content_b:?}"
        );
        assert_eq!(
            lower_a.contains("correct"),
            lower_b.contains("correct"),
            "both calls should reach the same verdict\n  A: {content_a:?}\n  B: {content_b:?}"
        );
        assert!(
            reasoning_a.is_some(),
            "reasoning should be present when thinking is enabled (call A)"
        );
        assert!(
            reasoning_b.is_some(),
            "reasoning should be present when thinking is enabled (call B)"
        );
    }

    /// Realistic end-to-end test using the exact same prompt format as
    /// `judge_answer_task_with_url`.  The model answer and reference answer are
    /// semantically equivalent but phrased differently ("x equals 5" vs "five"),
    /// creating real ambiguity.  Reports the full model output, then verifies
    /// the judge returns true.
    #[tokio::test]
    async fn test_realistic_ambiguous_judgment() {
        let _ = dotenvy::dotenv();
        let client = Client::new();

        let question = "What is the value of x if 3x + 5 = 20?";
        let model_answer = "x equals 5";
        let correct_answer = "five";

        // Exact same prompt construction as judge_answer_task_with_url
        let base_prompt = format!(
            "You are an answer checker that checks a model's answer against the reference answer. \
             Judge if the model's answer is equivalent to the reference answer. \
             Do not attempt to solve the problem yourself, only judge whether the given answer and the reference answer is equivalent. \
             If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
             The question is: \"{}\". \
             The model's answer is: \"{}\", and the correct answer is: \"{}\".",
            question, model_answer, correct_answer
        );
        let thinking_prompt = format!(
            "{base_prompt} Think step by step about whether the model's answer matches the reference answer. Put your final answer in \\boxed{{correct}} or \\boxed{{incorrect}}."
        );

        // ── Thinking mode (temp=0, thinking enabled) ──
        eprintln!("\n========== THINKING MODE (temp=0) ==========");
        let (content, reasoning) = call_api(&client, &thinking_prompt, 0.0, true).await;
        eprintln!("[content]\n{content}");
        eprintln!("[reasoning] {:?}", reasoning);
        let verdict = extract_boxed_verdict(&content);
        eprintln!("[extracted verdict from \\boxed{{}}] {:?}", verdict);
        assert!(
            verdict.is_some(),
            "thinking mode should produce a \\boxed{{}} response"
        );

        // ── Full judge pipeline ──
        let result = call_judge(&client, model_answer, correct_answer, question).await;
        eprintln!("\n[judge_answer_task_with_url result] {result}");
        assert!(
            result,
            "judge should recognize 'x equals 5' and 'five' as equivalent"
        );
    }
}
