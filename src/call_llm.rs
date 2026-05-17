use async_trait::async_trait;
use reqwest::Client;
use std::marker::PhantomData;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Semaphore;

use crate::llm_model::LlmModel;
use crate::llm_models::{
    LlmModelMarker, Qwen25, Qwen25TokenArray, Qwen35_4B, Qwen35TokenArray, Qwen3TokenArray,
    Qwen3_4B, Qwen3_8B,
};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
pub const CONTEXT_LENGTH_EXCEEDED_RESPONSE: &str = "<error>QWEN_CONTEXT_LENGTH_EXCEEDED</error>";

#[async_trait]
pub trait LlmCallable<M: LlmModelMarker>: Clone + Send + Sync {
    async fn generate(
        &self,
        prompt_or_tokens: M::StringOrTokenArray,
        passes_in_stop: bool,
    ) -> String;
}

pub async fn call_llm_with_prefix<M: LlmModelMarker, C: LlmCallable<M>>(
    llm_callable: &C,
    prompt_before_assistant: String,
    prompt_after_assistant: String,
) -> String {
    let prompt = M::build_prefix_thinking_disabled(&prompt_before_assistant, &prompt_after_assistant);
    let input = M::tokenize(prompt);
    llm_callable.generate(input, true).await
}

pub struct GptLlmCallable<M: LlmModelMarker> {
    client: Client,
    _marker: PhantomData<M>,
}

impl<M: LlmModelMarker> Clone for GptLlmCallable<M> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            _marker: PhantomData,
        }
    }
}

impl<M: LlmModelMarker<StringOrTokenArray = String>> GptLlmCallable<M> {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<M: LlmModelMarker<StringOrTokenArray = String>> LlmCallable<M> for GptLlmCallable<M> {
    async fn generate(
        &self,
        prompt_or_tokens: M::StringOrTokenArray,
        passes_in_stop: bool,
    ) -> String {
        let prompt = prompt_or_tokens;
        let body = if passes_in_stop {
            serde_json::json!({
                "model": M::API_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2048,
                "stop": ["</tool_wait>"],
            })
        } else {
            serde_json::json!({
                "model": M::API_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2048,
            })
        };

        let response = self
            .post_json(OPENAI_CHAT_COMPLETIONS_URL, body)
            .await
            .bytes()
            .await
            .unwrap();
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&response) else {
            panic!(
                "Failed to parse LLM response as JSON. Response text: {:?}",
                String::from_utf8_lossy(&response)
            );
        };
        json["choices"][0]["message"]["content"]
            .as_str()
            .expect(&format!("LLM response is invalid: {:?}", json))
            .to_string()
    }
}

#[derive(Clone)]
struct SharedQwenLlmCallable {
    client: Client,
    endpoint: Arc<LlmEndpoint>,
    api_name: &'static str,
}

impl SharedQwenLlmCallable {
    fn new(
        client: Client,
        api_name: &'static str,
        vllm_port: u16,
        max_concurrent_requests: usize,
    ) -> Self {
        Self {
            client,
            endpoint: Arc::new(LlmEndpoint::new(0, vllm_port, max_concurrent_requests)),
            api_name,
        }
    }

    async fn generate_from_tokens(&self, tokens: Vec<i32>, passes_in_stop: bool) -> String {
        let endpoint = self.endpoint.clone();
        let _permit = endpoint
            .request_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        let mut body = serde_json::json!({
            "model": self.api_name,
            "prompt_token_ids": tokens,
            "max_tokens": 2048,
            "include_stop_str_in_output": true,
        });
        if passes_in_stop {
            body["stop"] = serde_json::json!(["</tool_wait>"]);
        }

        let json: serde_json::Value = self
            .post_json(&endpoint.vllm_raw_completions_url(), body)
            .await
            .json()
            .await
            .unwrap();
        if let Some(error_message) = json["error"]["message"].as_str() {
            if error_message.contains("maximum context length")
                || error_message.contains("Please reduce the length of the input prompt")
                || error_message.contains("parameter=input_tokens")
            {
                return CONTEXT_LENGTH_EXCEEDED_RESPONSE.to_string();
            }
        }
        let content = json["choices"][0]["text"]
            .as_str()
            .expect(&format!(
                "Qwen completions response is invalid on endpoint {} (port {}): {:?}",
                endpoint.id, endpoint.vllm_port, json
            ))
            .to_string();
        endpoint.completed_requests.fetch_add(1, Ordering::SeqCst);
        content
    }

    async fn post_json(&self, url: &str, body: serde_json::Value) -> reqwest::Response {
        self.client.post(url).json(&body).send().await.unwrap()
    }
}

#[derive(Clone)]
pub struct Qwen25LlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen25LlmCallable {
    pub fn new(client: Client, vllm_port: u16, max_concurrent_requests: usize) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(
                client,
                Qwen25::API_NAME,
                vllm_port,
                max_concurrent_requests,
            ),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen25> for Qwen25LlmCallable {
    async fn generate(
        &self,
        prompt_or_tokens: Qwen25TokenArray,
        passes_in_stop: bool,
    ) -> String {
        self.shared
            .generate_from_tokens(prompt_or_tokens.tokens, passes_in_stop)
            .await
    }
}

#[derive(Clone)]
pub struct Qwen3LlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen3LlmCallable {
    pub fn new(client: Client, api_name: &'static str, vllm_port: u16, max_concurrent_requests: usize) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(client, api_name, vllm_port, max_concurrent_requests),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen3_4B> for Qwen3LlmCallable {
    async fn generate(
        &self,
        prompt_or_tokens: Qwen3TokenArray,
        passes_in_stop: bool,
    ) -> String {
        self.shared
            .generate_from_tokens(prompt_or_tokens.tokens, passes_in_stop)
            .await
    }
}

#[async_trait]
impl LlmCallable<Qwen3_8B> for Qwen3LlmCallable {
    async fn generate(
        &self,
        prompt_or_tokens: Qwen3TokenArray,
        passes_in_stop: bool,
    ) -> String {
        self.shared
            .generate_from_tokens(prompt_or_tokens.tokens, passes_in_stop)
            .await
    }
}

#[derive(Clone)]
pub struct Qwen35LlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen35LlmCallable {
    pub fn new(client: Client, vllm_port: u16, max_concurrent_requests: usize) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(
                client,
                Qwen35_4B::API_NAME,
                vllm_port,
                max_concurrent_requests,
            ),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen35_4B> for Qwen35LlmCallable {
    async fn generate(
        &self,
        prompt_or_tokens: Qwen35TokenArray,
        passes_in_stop: bool,
    ) -> String {
        self.shared
            .generate_from_tokens(prompt_or_tokens.tokens, passes_in_stop)
            .await
    }
}

impl<M: LlmModelMarker<StringOrTokenArray = String>> GptLlmCallable<M> {
    async fn post_json(&self, url: &str, body: serde_json::Value) -> reqwest::Response {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
        self.client
            .post(url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .unwrap()
    }
}

#[derive(Debug)]
pub struct LlmEndpoint {
    pub id: usize,
    pub vllm_port: u16,
    pub max_concurrent_requests: usize,
    pub question_slot_semaphore: Arc<Semaphore>,
    pub request_semaphore: Arc<Semaphore>,
    pub completed_requests: Arc<AtomicUsize>,
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
            max_concurrent_requests,
            question_slot_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            completed_requests: Arc::new(AtomicUsize::new(0)),
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
    let _permit = endpoint
        .request_semaphore
        .clone()
        .acquire_owned()
        .await
        .unwrap();
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
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!(
            "LLM response is invalid on endpoint {} (port {}): {:?}",
            endpoint.id, endpoint.vllm_port, json
        ))
        .to_string();
    endpoint.completed_requests.fetch_add(1, Ordering::SeqCst);
    content
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
    let _permit = endpoint
        .request_semaphore
        .clone()
        .acquire_owned()
        .await
        .unwrap();
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
    let content = json["choices"][0]["text"]
        .as_str()
        .expect(&format!(
            "Qwen completions response is invalid on endpoint {} (port {}): {:?}",
            endpoint.id, endpoint.vllm_port, json
        ))
        .to_string();
    endpoint.completed_requests.fetch_add(1, Ordering::SeqCst);
    content
}
