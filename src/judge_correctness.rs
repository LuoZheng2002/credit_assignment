use std::sync::{Arc, atomic::AtomicUsize};

use reqwest::Client;
use research_utility::progress_tui_logger::log_warning;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::{Duration, sleep};

use crate::{atomic_count_guard::AtomicCountGuard, direct_tool::trajectory::FinalAnswer};

const OPENROUTER_CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const JUDGE_DETERMINISTIC_SEED: i64 = 42;
const OPENROUTER_GPT4O_MODEL: &str = "openai/gpt-4o";
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
            JudgeAnswerModel::Gpt4o => OPENROUTER_GPT4O_MODEL,
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
    judge_answer_task_with_url(
        model_answer,
        correct_answer,
        question,
        client,
        OPENROUTER_CHAT_COMPLETIONS_URL,
    )
    .await
}

async fn judge_answer_task_with_url(
    model_answer: String,
    correct_answer: String,
    question: String,
    client: Client,
    url: &str,
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
            let fetch_result =
                fetch_judge_evaluation_with_url(&client, &prompt, model_to_try, url).await;
            match fetch_result {
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
                "Judger returned invalid response with model {}, attempt {}/{}. Last error: {}",
                model_to_try.display_name(),
                attempt,
                attempts_per_model,
                last_error
                    .as_deref()
                    .unwrap_or("none (response did not include 'correct' or 'incorrect')")
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

async fn fetch_judge_evaluation_with_url(
    client: &Client,
    prompt: &str,
    judge_model: JudgeAnswerModel,
    url: &str,
) -> Result<String, String> {
    let model_name = match judge_model {
        JudgeAnswerModel::Gpt4o => OPENROUTER_GPT4O_MODEL,
        JudgeAnswerModel::Gemini25FlashLite => OPENROUTER_GEMINI_25_FLASH_LITE_MODEL,
        JudgeAnswerModel::Gemini25Flash => OPENROUTER_GEMINI_25_FLASH_MODEL,
        JudgeAnswerModel::Gpt41Mini => OPENROUTER_GPT_41_MINI_MODEL,
    };
    let api_key_env = "OPENROUTER_API_KEY";
    let auth_is_bearer = true;

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
    };

    fn response_body() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "choices": [
                {
                    "message": {
                        "content": "correct"
                    }
                }
            ]
        }))
        .expect("response body should serialize")
    }

    async fn read_http_request(stream: &mut TcpStream) -> Result<(String, Vec<u8>), String> {
        let mut request = Vec::new();
        let mut buffer = [0u8; 1024];
        let header_end = loop {
            let read = stream
                .read(&mut buffer)
                .await
                .map_err(|err| format!("failed to read request: {err}"))?;
            if read == 0 {
                return Err("connection closed before request completed".to_string());
            }
            request.extend_from_slice(&buffer[..read]);
            if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break index;
            }
        };

        let headers = String::from_utf8(request[..header_end].to_vec())
            .map_err(|err| format!("request headers were not valid UTF-8: {err}"))?;
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.trim().eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .ok_or_else(|| "missing content-length header".to_string())?;
        let body_start = header_end + 4;
        let required_len = body_start + content_length;
        while request.len() < required_len {
            let read = stream
                .read(&mut buffer)
                .await
                .map_err(|err| format!("failed to read request body: {err}"))?;
            if read == 0 {
                return Err("connection closed before request body completed".to_string());
            }
            request.extend_from_slice(&buffer[..read]);
        }

        Ok((headers, request[body_start..required_len].to_vec()))
    }

    fn validate_request(headers: &str, body: &[u8], expected_model: &str, expected_api_key: &str) {
        let request_line = headers.lines().next().unwrap_or_default().to_lowercase();
        assert!(
            request_line.starts_with("post /api/v1/chat/completions http/1.1"),
            "unexpected request line: {}",
            request_line
        );
        assert!(
            headers
                .lines()
                .any(|line| line.to_lowercase().starts_with("authorization:")),
            "missing authorization header"
        );
        let authorization_line = headers
            .lines()
            .find(|line| line.to_lowercase().starts_with("authorization:"))
            .expect("authorization header should be present");
        assert_eq!(
            authorization_line
                .split_once(':')
                .expect("authorization header should contain colon")
                .1
                .trim(),
            format!("Bearer {}", expected_api_key)
        );
        assert!(
            headers.lines().any(|line| line
                .eq_ignore_ascii_case("HTTP-Referer: https://github.com/luoz/credit_assignment")),
            "missing HTTP-Referer header"
        );
        assert!(
            headers
                .lines()
                .any(|line| line.eq_ignore_ascii_case("X-Title: credit_assignment")),
            "missing X-Title header"
        );

        let parsed_body: Value = serde_json::from_slice(body).expect("body should be valid JSON");
        assert_eq!(parsed_body["model"], expected_model);
        assert_eq!(parsed_body["messages"][0]["role"], "user");
    }

    async fn spawn_mock_judge_server(
        expected_models: Vec<&'static str>,
        expected_api_key: String,
    ) -> (String, tokio::task::JoinHandle<Result<(), String>>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock server should bind");
        let addr = listener
            .local_addr()
            .expect("listener should have local addr");
        let url = format!("http://{}/api/v1/chat/completions", addr);
        let handle = tokio::spawn(async move {
            for expected_model in expected_models {
                let (mut stream, _) = listener
                    .accept()
                    .await
                    .map_err(|err| format!("failed to accept request: {err}"))?;
                let (headers, body) = read_http_request(&mut stream).await?;
                validate_request(&headers, &body, expected_model, &expected_api_key);
                let response_body = response_body();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.len()
                );
                stream
                    .write_all(response.as_bytes())
                    .await
                    .map_err(|err| format!("failed to write response headers: {err}"))?;
                stream
                    .write_all(&response_body)
                    .await
                    .map_err(|err| format!("failed to write response body: {err}"))?;
                stream
                    .shutdown()
                    .await
                    .map_err(|err| format!("failed to shutdown response stream: {err}"))?;
            }
            Ok(())
        });
        (url, handle)
    }

    #[tokio::test]
    async fn all_judge_models_use_openrouter_and_parse_correct_response() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY should be loaded from .env for this test");
        let expected_models = vec![
            JudgeAnswerModel::Gpt4o.display_name(),
            JudgeAnswerModel::Gemini25FlashLite.display_name(),
            JudgeAnswerModel::Gemini25Flash.display_name(),
            JudgeAnswerModel::Gpt41Mini.display_name(),
        ];
        let (url, server_handle) = spawn_mock_judge_server(expected_models, api_key).await;
        let client = Client::new();
        for judge_model in [
            JudgeAnswerModel::Gpt4o,
            JudgeAnswerModel::Gemini25FlashLite,
            JudgeAnswerModel::Gemini25Flash,
            JudgeAnswerModel::Gpt41Mini,
        ] {
            let evaluation = fetch_judge_evaluation_with_url(
                &client,
                "the question does not matter in this mock test",
                judge_model,
                &url,
            )
            .await
            .expect("mock judge request should succeed");
            assert_eq!(evaluation, "correct");
        }
        server_handle
            .await
            .expect("server task should join successfully")
            .expect("mock server should complete successfully");
    }

    #[tokio::test]
    async fn judge_answer_task_returns_true_for_correct_result() {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY should be loaded from .env for this test");
        let (url, server_handle) = spawn_mock_judge_server(
            vec![JudgeAnswerModel::Gemini25FlashLite.display_name()],
            api_key,
        )
        .await;
        let client = Client::new();
        let result = judge_answer_task_with_url(
            "42".to_string(),
            "42".to_string(),
            "what is six times seven?".to_string(),
            client,
            &url,
        )
        .await;
        assert!(result, "mock judge response should evaluate as correct");
        server_handle
            .await
            .expect("server task should join successfully")
            .expect("mock server should complete successfully");
    }
}
