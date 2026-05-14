use reqwest::Client;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::apply_vllm_model_chat_template::apply_vllm_model_chat_template;
use crate::direct_answer::generate_raw_answers::LlmModel;

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
pub const CONTEXT_LENGTH_EXCEEDED_RESPONSE: &str = "<error>QWEN_CONTEXT_LENGTH_EXCEEDED</error>";

#[derive(Debug)]
pub struct LlmEndpoint {
    pub id: usize,
    pub vllm_port: u16,
    pub question_slot_semaphore: Arc<Semaphore>,
    pub request_semaphore: Arc<Semaphore>,
}

impl LlmEndpoint {
    pub fn new(id: usize, vllm_port: u16, max_concurrent_requests: usize) -> Self {
        assert!(vllm_port > 0, "vLLM port must be greater than 0");
        assert!(
            max_concurrent_requests > 0,
            "max concurrent requests must be greater than 0"
        );
        Self {
            id,
            vllm_port,
            question_slot_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
        }
    }

    fn vllm_chat_completions_url(&self) -> String {
        format!("http://localhost:{}/v1/chat/completions", self.vllm_port)
    }

    fn vllm_raw_completions_url(&self) -> String {
        format!("http://localhost:{}/v1/completions", self.vllm_port)
    }
}

fn get_chat_completions_url(model: LlmModel) -> String {
    if model.is_gpt() {
        OPENAI_CHAT_COMPLETIONS_URL.to_string()
    } else if model.is_qwen() {
        panic!(
            "Qwen calls require an explicit endpoint. Use call_llm_chat_completions_on_endpoint(...)"
        )
    } else {
        panic!("Unsupported model family for {}", model.api_name());
    }
}

fn get_chat_completions_url_for_endpoint(model: LlmModel, endpoint: &LlmEndpoint) -> String {
    if model.is_gpt() {
        OPENAI_CHAT_COMPLETIONS_URL.to_string()
    } else if model.is_qwen() {
        endpoint.vllm_chat_completions_url()
    } else {
        panic!("Unsupported model family for {}", model.api_name());
    }
}

fn build_chat_completions_body(
    prompt: String,
    model: LlmModel,
    passes_in_stop: bool,
) -> serde_json::Value {
    if model.is_qwen() {
        // Explicitly disable Qwen3 thinking mode in vLLM chat templating.
        return serde_json::json!({
            "model": model.api_name(),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_completion_tokens": 2048,
            "stop": ["</tool_wait>"],
            "chat_template_kwargs": {
                "enable_thinking": false,
            }
        });
    } else if passes_in_stop {
        serde_json::json!({
            "model": model.api_name(),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_completion_tokens": 2048,
            "stop": ["</tool_wait>"],
        })
    } else {
        serde_json::json!({
            "model": model.api_name(),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_completion_tokens": 2048,
        })
    }
}

async fn post_json(
    client: Client,
    url: &str,
    body: serde_json::Value,
    model: LlmModel,
) -> reqwest::Response {
    if model.is_gpt() {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
        return client
            .post(url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .unwrap();
    }

    client.post(url).json(&body).send().await.unwrap()
}

pub async fn call_llm_chat_completions(
    client: Client,
    prompt: String,
    model: LlmModel,
    passes_in_stop: bool,
) -> String {
    let url = get_chat_completions_url(model);
    let body = build_chat_completions_body(prompt, model, passes_in_stop);
    let response = post_json(client, &url, body, model).await;
    let body = response.bytes().await.unwrap();
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) else {
        panic!(
            "Failed to parse LLM response as JSON. Response text: {:?}",
            String::from_utf8_lossy(&body)
        );
    };
    json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("LLM response is invalid: {:?}", json))
        .to_string()
}

pub async fn call_llm_chat_completions_on_endpoint(
    client: Client,
    prompt: String,
    model: LlmModel,
    passes_in_stop: bool,
    endpoint: Arc<LlmEndpoint>,
) -> String {
    let _permit = endpoint.request_semaphore.clone().acquire_owned().await.unwrap();
    let url = get_chat_completions_url_for_endpoint(model, &endpoint);
    let body = build_chat_completions_body(prompt, model, passes_in_stop);
    let response = post_json(client, &url, body, model).await;
    let body = response.bytes().await.unwrap();
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) else {
        panic!(
            "Failed to parse LLM response as JSON on endpoint {} (port {}). Response text: {:?}",
            endpoint.id,
            endpoint.vllm_port,
            String::from_utf8_lossy(&body)
        );
    };
    json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!(
            "LLM response is invalid on endpoint {} (port {}): {:?}",
            endpoint.id, endpoint.vllm_port, json
        ))
        .to_string()
}

pub async fn call_qwen_raw_completions_on_endpoint(
    client: Client,
    chat_template_prompt: String,
    model: LlmModel,
    endpoint: Arc<LlmEndpoint>,
) -> String {
    assert!(
        model.is_qwen(),
        "call_qwen_raw_completions_on_endpoint only supports Qwen-family models",
    );
    let _permit = endpoint.request_semaphore.clone().acquire_owned().await.unwrap();
    let body = serde_json::json!({
        "model": model.api_name(),
        "prompt": chat_template_prompt,
        "max_tokens": 2048,
        "stop": ["</tool_wait>"],
        "include_stop_str_in_output": true,
    });

    let response = post_json(client, &endpoint.vllm_raw_completions_url(), body, model).await;
    let json: serde_json::Value = response.json().await.unwrap();
    if let Some(error_message) = json["error"]["message"].as_str() {
        if error_message.contains("maximum context length")
            || error_message.contains("Please reduce the length of the input prompt")
            || error_message.contains("parameter=input_tokens")
        {
            return CONTEXT_LENGTH_EXCEEDED_RESPONSE.to_string();
        }
    }
    json["choices"][0]["text"]
        .as_str()
        .expect(&format!(
            "Qwen completions response is invalid on endpoint {} (port {}): {:?}",
            endpoint.id, endpoint.vllm_port, json
        ))
        .to_string()
}

pub async fn call_llm_with_prefix_on_endpoint(
    client: Client,
    prompt_before_assistant: String,
    prompt_after_assistant: String,
    model: LlmModel,
    endpoint: Arc<LlmEndpoint>,
) -> String {
    if model.is_qwen() {
        let mut planner_chat_template_prompt =
            apply_vllm_model_chat_template(model, &prompt_before_assistant, false);
        planner_chat_template_prompt += &prompt_after_assistant;
        call_qwen_raw_completions_on_endpoint(
            client.clone(),
            planner_chat_template_prompt,
            model,
            endpoint,
        )
        .await
    } else {
        let full_prompt = format!(
            "{}\nAssistant: {}",
            prompt_before_assistant, prompt_after_assistant
        );
        call_llm_chat_completions_on_endpoint(client.clone(), full_prompt, model, true, endpoint)
            .await
    }
}
