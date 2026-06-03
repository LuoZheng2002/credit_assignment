use std::sync::{Arc, atomic::AtomicUsize};

use reqwest::Client;
use research_utility::progress_tui_server::log_warning;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::{Duration, sleep};

use crate::{atomic_count_guard::AtomicCountGuard, direct_tool::direct_trajectory::FinalAnswer};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
const OPENROUTER_CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPENAI_GPT4O_MODEL: &str = "gpt-4o";
const OPENROUTER_DEEPSEEK_V4_FLASH_MODEL: &str = "deepseek/deepseek-v4-flash";

#[derive(Clone, Copy)]
pub enum JudgeAnswerModel {
    Gpt4o,
    DeepseekV4Flash,
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
    judge_model: JudgeAnswerModel,
) -> bool {
    let prompt = format!(
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
The question is: \"{}\". \
The model's answer is: \"{}\", and the correct answer is: \"{}\". Return only 'correct' or 'incorrect'.",
        question, model_answer, correct_answer
    );
    let mut last_error: Option<String> = None;

    let num_attempts: usize = 20;

    for attempt in 0..=num_attempts {
        match fetch_judge_evaluation(&client, &prompt, judge_model).await {
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
            "Judger returned invalid response, attempt {}",
            attempt
        ));

        if attempt < num_attempts {
            sleep(Duration::from_secs(1)).await;
        }
    }

    panic!(
        "Failed to judge answer after {} attempts: {}",
        num_attempts,
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
        JudgeAnswerModel::DeepseekV4Flash => (
            OPENROUTER_CHAT_COMPLETIONS_URL,
            OPENROUTER_DEEPSEEK_V4_FLASH_MODEL,
            "OPENROUTER_API_KEY",
            true,
        ),
    };

    let api_key = std::env::var(api_key_env)
        .map_err(|_| format!("{api_key_env} environment variable not set"))?;
    let mut body = serde_json::json!({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 32,
        "temperature": 0.0,
    });
    if matches!(judge_model, JudgeAnswerModel::DeepseekV4Flash) {
        body["reasoning"] = serde_json::json!({
            "enabled": false
        });
    }

    let mut request_builder = client.post(url).json(&body);
    if auth_is_bearer {
        request_builder = request_builder.bearer_auth(api_key);
    }
    if matches!(judge_model, JudgeAnswerModel::DeepseekV4Flash) {
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
    judge_model: JudgeAnswerModel,
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
                judge_model,
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
