use reqwest::Client;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokenizers::Tokenizer;
use tokio::sync::Semaphore;

pub const CONTEXT_LENGTH_EXCEEDED_RESPONSE: &str = "<error>QWEN_CONTEXT_LENGTH_EXCEEDED</error>";

pub(crate) fn encode_to_i32_ids(tokenizer: &Tokenizer, text: &str) -> Vec<i32> {
    tokenizer
        .encode(text, false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|token| i32::try_from(*token).expect("token id must fit in i32"))
        .collect()
}

pub(crate) fn decode_from_i32_ids(tokenizer: &Tokenizer, token_ids: &[i32]) -> String {
    let token_ids: Vec<u32> = token_ids
        .iter()
        .map(|token| u32::try_from(*token).expect("token id must be non-negative"))
        .collect();
    tokenizer
        .decode(&token_ids, false)
        .expect("failed to decode token ids")
}

pub(crate) fn token_to_i32_id(tokenizer: &Tokenizer, token: &str, api_name: &str) -> i32 {
    let token_id = tokenizer
        .token_to_id(token)
        .unwrap_or_else(|| panic!("Token '{token}' must exist in {} tokenizer", api_name));
    i32::try_from(token_id).expect("token id must fit in i32")
}

#[derive(Clone)]
pub(crate) struct SharedQwenLlmCallable {
    client: Client,
    endpoint: Arc<LlmEndpoint>,
    api_name: &'static str,
}

impl SharedQwenLlmCallable {
    pub(crate) fn new(
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

    pub(crate) async fn generate_from_tokens(
        &self,
        tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> String {
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
            .unwrap_or_else(|| {
                panic!(
                    "Qwen completions response is invalid on endpoint {} (port {}): {:?}",
                    endpoint.id, endpoint.vllm_port, json
                )
            })
            .to_string();
        endpoint.completed_requests.fetch_add(1, Ordering::SeqCst);
        content
    }

    async fn post_json(&self, url: &str, body: serde_json::Value) -> reqwest::Response {
        self.client.post(url).json(&body).send().await.unwrap()
    }
}

#[derive(Debug)]
struct LlmEndpoint {
    id: usize,
    vllm_port: u16,
    request_semaphore: Arc<Semaphore>,
    completed_requests: Arc<AtomicUsize>,
}

impl LlmEndpoint {
    fn new(id: usize, vllm_port: u16, max_concurrent_requests: usize) -> Self {
        assert!(vllm_port > 0, "vLLM port must be greater than 0");
        assert!(
            max_concurrent_requests > 0,
            "max concurrent requests must be greater than 0"
        );
        Self {
            id,
            vllm_port,
            request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            completed_requests: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn vllm_raw_completions_url(&self) -> String {
        format!("http://localhost:{}/v1/completions", self.vllm_port)
    }
}
