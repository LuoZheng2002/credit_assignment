use reqwest::Client;
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokenizers::Tokenizer;
use tokio::sync::Semaphore;

use crate::token_array::{TokenArrayWithLogprob, TokenLogprobCandidate};

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

#[derive(Clone, Debug)]
pub(crate) enum QwenBackend {
    Vllm {
        vllm_port: u16,
    },
    OpenRouter {
        base_url: String,
        model: String,
        api_key: String,
        http_referer: Option<String>,
        x_title: Option<String>,
    },
}

#[derive(Clone)]
pub(crate) struct SharedQwenLlmCallable {
    client: Client,
    backend: QwenBackend,
    api_name: &'static str,
    decode_tokens: fn(&[i32]) -> String,
    encode_text: fn(&str) -> Vec<i32>,
    request_semaphore: Arc<Semaphore>,
    completed_requests: Arc<AtomicUsize>,
}

impl SharedQwenLlmCallable {
    pub(crate) fn new(
        client: Client,
        api_name: &'static str,
        backend: QwenBackend,
        max_concurrent_requests: usize,
        decode_tokens: fn(&[i32]) -> String,
        encode_text: fn(&str) -> Vec<i32>,
    ) -> Self {
        assert!(
            max_concurrent_requests > 0,
            "max concurrent requests must be greater than 0"
        );
        if let QwenBackend::Vllm { vllm_port } = backend {
            assert!(vllm_port > 0, "vLLM port must be greater than 0");
        }

        Self {
            client,
            backend,
            api_name,
            decode_tokens,
            encode_text,
            request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            completed_requests: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(crate) async fn generate_from_tokens(
        &self,
        tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> String {
        let _permit = self.request_semaphore.clone().acquire_owned().await.unwrap();

        let response = match &self.backend {
            QwenBackend::Vllm { vllm_port } => {
                let mut body = serde_json::json!({
                    "model": self.api_name,
                    "prompt_token_ids": tokens,
                    "max_tokens": 2048,
                    "include_stop_str_in_output": true,
                });
                if passes_in_stop {
                    body["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                self.post_json(&format!("http://localhost:{vllm_port}/v1/completions"), body)
                    .await
            }
            QwenBackend::OpenRouter {
                base_url,
                model,
                api_key,
                http_referer,
                x_title,
            } => {
                let prompt = (self.decode_tokens)(&tokens);
                let mut body = serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                });
                if passes_in_stop {
                    body["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                self.post_openrouter_json(base_url, api_key, http_referer, x_title, body)
                    .await
            }
        };

        if let Some(error_message) = response["error"]["message"].as_str() {
            if error_message.contains("maximum context length")
                || error_message.contains("Please reduce the length of the input prompt")
                || error_message.contains("parameter=input_tokens")
                || error_message.contains("context length")
            {
                return CONTEXT_LENGTH_EXCEEDED_RESPONSE.to_string();
            }
        }

        let content = match &self.backend {
            QwenBackend::Vllm { vllm_port } => response["choices"][0]["text"]
                .as_str()
                .unwrap_or_else(|| {
                    panic!(
                        "Qwen vLLM response is invalid (port {}): {:?}",
                        vllm_port, response
                    )
                })
                .to_string(),
            QwenBackend::OpenRouter { .. } => parse_openrouter_message_content(&response)
                .unwrap_or_else(|| {
                    panic!("Qwen OpenRouter response is invalid: {:?}", response)
                }),
        };

        self.completed_requests.fetch_add(1, Ordering::SeqCst);
        content
    }

    pub(crate) async fn generate_tokens_with_logprobs_from_tokens(
        &self,
        tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> TokenArrayWithLogprob {
        let _permit = self.request_semaphore.clone().acquire_owned().await.unwrap();

        let result = match &self.backend {
            QwenBackend::Vllm { vllm_port } => {
                let mut body = serde_json::json!({
                    "model": self.api_name,
                    "prompt_token_ids": tokens,
                    "max_tokens": 2048,
                    "include_stop_str_in_output": true,
                    "logprobs": 8,
                    "return_tokens_as_token_ids": true,
                });
                if passes_in_stop {
                    body["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                let json = self
                    .post_json(&format!("http://localhost:{vllm_port}/v1/completions"), body)
                    .await;
                parse_vllm_response_with_logprobs(vllm_port, &json)
            }
            QwenBackend::OpenRouter {
                base_url,
                model,
                api_key,
                http_referer,
                x_title,
            } => {
                let prompt = (self.decode_tokens)(&tokens);
                let mut body = serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "logprobs": true,
                    "top_logprobs": 8,
                    "provider": {
                        "require_parameters": true
                    }
                });
                if passes_in_stop {
                    body["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                let json = self
                    .post_openrouter_json(base_url, api_key, http_referer, x_title, body)
                    .await;
                parse_openrouter_response_with_logprobs(&json, self.encode_text)
            }
        };

        self.completed_requests.fetch_add(1, Ordering::SeqCst);
        result
    }

    async fn post_json(&self, url: &str, body: serde_json::Value) -> Value {
        self.client
            .post(url)
            .json(&body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn post_openrouter_json(
        &self,
        base_url: &str,
        api_key: &str,
        http_referer: &Option<String>,
        x_title: &Option<String>,
        body: serde_json::Value,
    ) -> Value {
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let mut request = self
            .client
            .post(url)
            .bearer_auth(api_key)
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(http_referer) = http_referer {
            request = request.header("HTTP-Referer", http_referer);
        }
        if let Some(x_title) = x_title {
            request = request.header("X-Title", x_title);
        }

        request.send().await.unwrap().json().await.unwrap()
    }
}

fn parse_vllm_response_with_logprobs(vllm_port: &u16, json: &Value) -> TokenArrayWithLogprob {
    if let Some(error_message) = json["error"]["message"].as_str() {
        panic!(
            "Qwen completion with logprobs failed on vLLM port {}: {}. Full response: {:?}",
            vllm_port, error_message, json
        );
    }

    let choice = &json["choices"][0];
    let decoded_string = choice["text"]
        .as_str()
        .unwrap_or_else(|| {
            panic!(
                "Qwen completion response missing choices[0].text on vLLM port {}: {:?}",
                vllm_port, json
            )
        })
        .to_string();

    let generated_tokens: Vec<i32> = choice["token_ids"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "Qwen completion response missing choices[0].token_ids on vLLM port {}: {:?}",
                vllm_port, json
            )
        })
        .iter()
        .map(|token| {
            i32::try_from(
                token
                    .as_i64()
                    .unwrap_or_else(|| panic!("token id must be i64-compatible: {token:?}")),
            )
            .expect("token id must fit in i32")
        })
        .collect();

    let token_logprobs = choice["logprobs"]["token_logprobs"].as_array();
    let top_logprobs = choice["logprobs"]["top_logprobs"].as_array();

    let mut aligned_logprobs = Vec::with_capacity(generated_tokens.len());
    for (idx, generated_token_id) in generated_tokens.iter().copied().enumerate() {
        let generated_logprob = token_logprobs
            .and_then(|vals| vals.get(idx))
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(f32::NEG_INFINITY);

        let mut candidates = top_logprobs
            .and_then(|vals| vals.get(idx))
            .map(parse_vllm_candidates)
            .unwrap_or_default();

        if !candidates
            .iter()
            .any(|candidate| candidate.token_id == generated_token_id)
        {
            candidates.push(TokenLogprobCandidate {
                token_id: generated_token_id,
                logprob: generated_logprob,
            });
        }

        candidates.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
        candidates.dedup_by(|a, b| a.token_id == b.token_id);

        let mut top8 = [TokenLogprobCandidate {
            token_id: generated_token_id,
            logprob: f32::NEG_INFINITY,
        }; 8];

        for (slot, candidate) in candidates.into_iter().take(8).enumerate() {
            top8[slot] = candidate;
        }
        aligned_logprobs.push(top8);
    }

    TokenArrayWithLogprob {
        tokens: generated_tokens,
        decoded_string,
        logprobs: aligned_logprobs,
    }
}

fn parse_openrouter_response_with_logprobs(
    json: &Value,
    encode_text: fn(&str) -> Vec<i32>,
) -> TokenArrayWithLogprob {
    if let Some(error_message) = json["error"]["message"].as_str() {
        panic!(
            "Qwen completion with logprobs failed on OpenRouter: {}. Full response: {:?}",
            error_message, json
        );
    }

    if json["choices"][0]["logprobs"].is_null() {
        panic!(
            "Qwen OpenRouter response returned null logprobs even though logprobs were requested. \
This usually means the selected provider/model route does not support logprobs. \
Try a different OpenRouter model/provider. Full response: {:?}",
            json
        );
    }

    let content_entries = json["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "Qwen OpenRouter response missing choices[0].logprobs.content; model may not support logprobs. Full response: {:?}",
                json
            )
        });

    let mut tokens = Vec::with_capacity(content_entries.len());
    let mut decoded_string = String::new();
    let mut logprobs = Vec::with_capacity(content_entries.len());

    for entry in content_entries {
        let sampled_token = entry["token"]
            .as_str()
            .unwrap_or_else(|| panic!("OpenRouter logprob entry missing token: {entry:?}"));
        let Some(sampled_token_id) = try_token_to_single_id(encode_text, sampled_token) else {
            continue;
        };

        decoded_string.push_str(sampled_token);
        tokens.push(sampled_token_id);
        let sampled_logprob = entry["logprob"].as_f64().unwrap_or(f64::NEG_INFINITY) as f32;

        let mut candidates: Vec<TokenLogprobCandidate> = entry["top_logprobs"]
            .as_array()
            .map(|top| {
                top.iter()
                    .filter_map(|candidate| {
                        let token = candidate["token"].as_str()?;
                        let token_id = try_token_to_single_id(encode_text, token)?;
                        let logprob = candidate["logprob"].as_f64().unwrap_or(f64::NEG_INFINITY)
                            as f32;
                        Some(TokenLogprobCandidate { token_id, logprob })
                    })
                    .collect()
            })
            .unwrap_or_default();

        if !candidates.iter().any(|c| c.token_id == sampled_token_id) {
            candidates.push(TokenLogprobCandidate {
                token_id: sampled_token_id,
                logprob: sampled_logprob,
            });
        }

        candidates.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
        candidates.dedup_by(|a, b| a.token_id == b.token_id);

        let mut top8 = [TokenLogprobCandidate {
            token_id: sampled_token_id,
            logprob: f32::NEG_INFINITY,
        }; 8];
        for (slot, candidate) in candidates.into_iter().take(8).enumerate() {
            top8[slot] = candidate;
        }
        logprobs.push(top8);
    }

    if tokens.is_empty() {
        panic!(
            "Qwen OpenRouter returned logprobs but no mappable tokens for tokenizer. Full response: {:?}",
            json
        );
    }

    TokenArrayWithLogprob {
        tokens,
        decoded_string,
        logprobs,
    }
}

fn parse_openrouter_message_content(json: &Value) -> Option<String> {
    let message_content = &json["choices"][0]["message"]["content"];
    if let Some(text) = message_content.as_str() {
        return Some(text.to_string());
    }

    let parts = message_content.as_array()?;
    let mut combined = String::new();
    for part in parts {
        if let Some(text) = part["text"].as_str() {
            combined.push_str(text);
        }
    }
    if combined.is_empty() {
        None
    } else {
        Some(combined)
    }
}

fn try_token_to_single_id(encode_text: fn(&str) -> Vec<i32>, token: &str) -> Option<i32> {
    let encoded = encode_text(token);
    if encoded.len() == 1 {
        Some(encoded[0])
    } else {
        None
    }
}

fn parse_vllm_candidates(value: &Value) -> Vec<TokenLogprobCandidate> {
    if let Some(array) = value.as_array() {
        return array
            .iter()
            .filter_map(|entry| {
                let map = entry.as_object()?;
                let token_id = map.get("token_id")?.as_i64()?;
                let logprob = map.get("logprob")?.as_f64()? as f32;
                Some(TokenLogprobCandidate {
                    token_id: i32::try_from(token_id).ok()?,
                    logprob,
                })
            })
            .collect();
    }

    panic!(
        "Unexpected vLLM top_logprobs shape; expected array entries with token_id/logprob. Got: {:?}",
        value
    )
}
