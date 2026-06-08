use std::sync::{Arc, atomic::AtomicUsize};

use reqwest::Client;
use research_utility::progress_tui_logger::log_warning;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::{Duration, sleep};

use crate::{atomic_count_guard::AtomicCountGuard, direct_tool::direct_trajectory::FinalAnswer};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
const OPENROUTER_CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const JUDGE_DETERMINISTIC_SEED: i64 = 42;
const OPENAI_GPT4O_MODEL: &str = "gpt-4o";
const OPENROUTER_GEMINI_25_FLASH_LITE_MODEL: &str = "google/gemini-2.5-flash-lite";
const OPENROUTER_GEMINI_25_FLASH_MODEL: &str = "google/gemini-2.5-flash";
const OPENROUTER_GPT_41_MINI_MODEL: &str = "openai/gpt-4.1-mini";

#[derive(Clone, Copy)]
pub enum JudgeAnswerModel {
    Gpt4o,
    Gemini25FlashLite,
    Gemini25Flash,
    Gpt41Mini,
}

impl JudgeAnswerModel {
    fn display_name(&self) -> &'static str {
        match self {
            JudgeAnswerModel::Gpt4o => OPENAI_GPT4O_MODEL,
            JudgeAnswerModel::Gemini25FlashLite => OPENROUTER_GEMINI_25_FLASH_LITE_MODEL,
            JudgeAnswerModel::Gemini25Flash => OPENROUTER_GEMINI_25_FLASH_MODEL,
            JudgeAnswerModel::Gpt41Mini => OPENROUTER_GPT_41_MINI_MODEL,
        }
    }
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
    let prompt = format!(
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
Do not attempt to solve the problem yourself, only judge whether the given answer and the reference answer is equivalent. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
The question is: \"{}\". \
The model's answer is: \"{}\", and the correct answer is: \"{}\". Return only 'correct' or 'incorrect'.",
        question, model_answer, correct_answer
    );
    let mut last_error: Option<String> = None;
    let attempts_per_model: usize = 1;
    let model_sequence = vec![
        JudgeAnswerModel::Gemini25FlashLite,
        JudgeAnswerModel::Gemini25Flash,
        JudgeAnswerModel::Gpt4o,
        JudgeAnswerModel::Gpt41Mini,
    ];

    let total_attempts = attempts_per_model * model_sequence.len();
    for (model_index, model_to_try) in model_sequence.into_iter().enumerate() {
        if model_index > 0 {
            log_warning(format!(
                "Falling back to judge model {} after previous model failures",
                model_to_try.display_name()
            ));
        }
        for attempt in 1..=attempts_per_model {
            match fetch_judge_evaluation(&client, &prompt, model_to_try).await {
                Ok(evaluation) => {
                    let evaluation = evaluation.trim().to_lowercase();
                    if evaluation.contains("incorrect") {
                        return false;
                    }
                    if evaluation.contains("correct") {
                        return true;
                    }
                    last_error = Some(format!("Unexpected evaluation result: {}", evaluation));
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                }
            }
            log_warning(format!(
                "Judger returned invalid response with model {}, attempt {}/{}",
                model_to_try.display_name(),
                attempt,
                attempts_per_model
            ));

            if attempt < attempts_per_model {
                sleep(Duration::from_secs(1)).await;
            }
        }
    }

    panic!(
        "Failed to judge answer after {} attempts across primary and fallback models: {}",
        total_attempts,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    );
}

async fn fetch_judge_evaluation(
    client: &Client,
    prompt: &str,
    judge_model: JudgeAnswerModel,
) -> Result<String, String> {
    let (url, model_name, api_key_env, auth_is_bearer) = match judge_model {
        JudgeAnswerModel::Gpt4o => (
            OPENAI_CHAT_COMPLETIONS_URL,
            OPENAI_GPT4O_MODEL,
            "OPENAI_API_KEY",
            true,
        ),
        JudgeAnswerModel::Gemini25FlashLite => (
            OPENROUTER_CHAT_COMPLETIONS_URL,
            OPENROUTER_GEMINI_25_FLASH_LITE_MODEL,
            "OPENROUTER_API_KEY",
            true,
        ),
        JudgeAnswerModel::Gemini25Flash => (
            OPENROUTER_CHAT_COMPLETIONS_URL,
            OPENROUTER_GEMINI_25_FLASH_MODEL,
            "OPENROUTER_API_KEY",
            true,
        ),
        JudgeAnswerModel::Gpt41Mini => (
            OPENROUTER_CHAT_COMPLETIONS_URL,
            OPENROUTER_GPT_41_MINI_MODEL,
            "OPENROUTER_API_KEY",
            true,
        ),
    };

    let api_key = std::env::var(api_key_env)
        .map_err(|_| format!("{api_key_env} environment variable not set"))?;
    let body = serde_json::json!({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 32,
        "temperature": 0.0,
        "seed": JUDGE_DETERMINISTIC_SEED,
    });

    let mut request_builder = client.post(url).json(&body);
    if auth_is_bearer {
        request_builder = request_builder.bearer_auth(api_key);
    }
    if matches!(
        judge_model,
        JudgeAnswerModel::Gemini25FlashLite
            | JudgeAnswerModel::Gemini25Flash
            | JudgeAnswerModel::Gpt41Mini
    ) {
        request_builder = request_builder
            .header("HTTP-Referer", "https://github.com/luoz/credit_assignment")
            .header("X-Title", "credit_assignment");
    }

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

    let content = &response_json["choices"][0]["message"]["content"];
    if let Some(evaluation) = content.as_str() {
        return Ok(evaluation.to_string());
    }
    if let Some(parts) = content.as_array() {
        let merged = parts
            .iter()
            .filter_map(|entry| entry["text"].as_str())
            .collect::<String>();
        if !merged.is_empty() {
            return Ok(merged);
        }
    }

    Err(format!("Judge response is invalid: {response_json:?}"))
}

pub async fn judge_final_answer(
    final_answer: &FinalAnswer,
    correct_answer: &str,
    question: &str,
    client: Client,
    _judge_model: JudgeAnswerModel,
    num_judge_waiting_workers: Arc<AtomicUsize>,
) -> CorrectnessJudgment {
    let is_correct = match final_answer {
        FinalAnswer::ModelProvided(model_answer) => {
            let _num_judge_waiting_workers_guard = AtomicCountGuard::new(
                num_judge_waiting_workers.clone(),
                "judge_waiting_workers".to_string(),
            );
            judge_answer_task(
                model_answer.clone(),
                correct_answer.to_string(),
                question.to_string(),
                client,
            )
            .await
        }
        FinalAnswer::Failure(_error_message) => false,
    };
    CorrectnessJudgment {
        model_answer: final_answer.clone(),
        correct_answer: correct_answer.to_string(),
        is_correct,
    }
}
