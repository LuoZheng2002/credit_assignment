use reqwest::Client;
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokenizers::Tokenizer;
use tokio::sync::Semaphore;
use tonic::Request;

use crate::token_array::{TokenArrayWithLogprob, TokenLogprobCandidate};
use crate::vllm_wrapper::{VllmPrompt, VllmRequest as WrapperRequest, VllmResponse, proto};

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
    Sglang {
        sglang_port: u16,
    },
    VllmWrapper {
        vllm_wrapper_port: u16,
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
        if let QwenBackend::Sglang { sglang_port } = backend {
            assert!(sglang_port > 0, "SGLang port must be greater than 0");
        }
        if let QwenBackend::VllmWrapper { vllm_wrapper_port } = backend {
            assert!(
                vllm_wrapper_port > 0,
                "vLLM wrapper port must be greater than 0"
            );
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
        let _permit = self
            .request_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        let response = match &self.backend {
            QwenBackend::Vllm { vllm_port } => {
                let prompt = (self.decode_tokens)(&tokens);
                let mut body = serde_json::json!({
                    "model": self.api_name,
                    "prompt": prompt,
                    "max_tokens": 2048,
                    "include_stop_str_in_output": true,
                });
                if passes_in_stop {
                    body["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                self.post_json(
                    &format!("http://localhost:{vllm_port}/v1/completions"),
                    body,
                )
                .await
            }
            QwenBackend::Sglang { sglang_port } => {
                let mut body = serde_json::json!({
                    "input_ids": tokens.clone(),
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 2048,
                    },
                    "stream": false,
                });
                if passes_in_stop {
                    body["sampling_params"]["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                self.post_json(&format!("http://localhost:{sglang_port}/generate"), body)
                    .await
            }
            QwenBackend::VllmWrapper { vllm_wrapper_port } => {
                return self
                    .generate_from_tokens_with_vllm_wrapper(
                        tokens,
                        *vllm_wrapper_port,
                        passes_in_stop,
                    )
                    .await;
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
            if is_context_length_error(error_message) {
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
            QwenBackend::Sglang { sglang_port } => {
                let generated_tokens = parse_sglang_generated_token_ids(
                    &response,
                    &tokens,
                    &format!("SGLang port {}", sglang_port),
                );
                (self.decode_tokens)(&generated_tokens)
            }
            QwenBackend::OpenRouter { .. } => parse_openrouter_message_content(&response)
                .unwrap_or_else(|| panic!("Qwen OpenRouter response is invalid: {:?}", response)),
            QwenBackend::VllmWrapper { .. } => {
                unreachable!("vLLM-wrapper path returns before JSON parsing")
            }
        };

        self.completed_requests.fetch_add(1, Ordering::SeqCst);
        content
    }

    pub(crate) async fn generate_tokens_with_logprobs_from_tokens(
        &self,
        tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
    ) -> TokenArrayWithLogprob {
        let _permit = self
            .request_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        let result = match &self.backend {
            QwenBackend::Vllm { vllm_port } => {
                let prompt = (self.decode_tokens)(&tokens);
                let mut body = serde_json::json!({
                    "model": self.api_name,
                    "prompt": prompt,
                    "max_tokens": 2048,
                    "temperature": temperature,
                    "include_stop_str_in_output": true,
                    "logprobs": 8,
                });
                if passes_in_stop {
                    body["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                let json = self
                    .post_json(
                        &format!("http://localhost:{vllm_port}/v1/completions"),
                        body,
                    )
                    .await;
                parse_completion_response_with_logprobs(
                    &format!("vLLM port {}", vllm_port),
                    &json,
                    self.encode_text,
                )
            }
            QwenBackend::Sglang { sglang_port } => {
                let mut body = serde_json::json!({
                    "input_ids": tokens.clone(),
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": 2048,
                    },
                    "return_logprob": true,
                    "logprob_start_len": -1,
                    "top_logprobs_num": 8,
                    "stream": false,
                });
                if passes_in_stop {
                    body["sampling_params"]["stop"] = serde_json::json!(["</tool_wait>"]);
                }

                let json = self
                    .post_json(&format!("http://localhost:{sglang_port}/generate"), body)
                    .await;
                parse_sglang_response_with_logprobs(
                    &format!("SGLang port {}", sglang_port),
                    &json,
                    &tokens,
                    self.decode_tokens,
                )
            }
            QwenBackend::VllmWrapper { vllm_wrapper_port } => {
                self.generate_tokens_with_logprobs_with_vllm_wrapper(
                    tokens,
                    *vllm_wrapper_port,
                    passes_in_stop,
                    temperature,
                )
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
                    "temperature": temperature,
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

    async fn generate_from_tokens_with_vllm_wrapper(
        &self,
        tokens: Vec<i32>,
        vllm_wrapper_port: u16,
        passes_in_stop: bool,
    ) -> String {
        let request = WrapperRequest {
            model_name: self.api_name.to_string(),
            prompt: VllmPrompt::TokenIds(tokens),
            max_tokens: 2048,
            include_stop_str_in_output: true,
            requires_logprobs: false,
            stop: if passes_in_stop {
                vec!["</tool_wait>".to_string()]
            } else {
                Vec::new()
            },
            temperature: 0.0,
        };

        let response = self
            .call_vllm_wrapper_with_retry(vllm_wrapper_port, request)
            .await;

        match response {
            VllmResponse::Success { response_text, .. } => response_text,
            VllmResponse::Error { error_message } => {
                if is_context_length_error(&error_message) {
                    CONTEXT_LENGTH_EXCEEDED_RESPONSE.to_string()
                } else {
                    panic!(
                        "Qwen vLLM-wrapper request failed after retries on port {}: {}",
                        vllm_wrapper_port, error_message
                    )
                }
            }
        }
    }

    async fn generate_tokens_with_logprobs_with_vllm_wrapper(
        &self,
        tokens: Vec<i32>,
        vllm_wrapper_port: u16,
        passes_in_stop: bool,
        temperature: f32,
    ) -> TokenArrayWithLogprob {
        let request = WrapperRequest {
            model_name: self.api_name.to_string(),
            prompt: VllmPrompt::TokenIds(tokens),
            max_tokens: 2048,
            include_stop_str_in_output: true,
            requires_logprobs: true,
            stop: if passes_in_stop {
                vec!["</tool_wait>".to_string()]
            } else {
                Vec::new()
            },
            temperature,
        };

        let response = self
            .call_vllm_wrapper_with_retry(vllm_wrapper_port, request)
            .await;

        response
            .into_token_array_with_logprob()
            .unwrap_or_else(|error| {
                panic!(
                    "Qwen vLLM-wrapper logprob request failed on port {}: {}",
                    vllm_wrapper_port, error
                )
            })
    }

    async fn call_vllm_wrapper_with_retry(
        &self,
        vllm_wrapper_port: u16,
        request: WrapperRequest,
    ) -> VllmResponse {
        let endpoint = format!("http://127.0.0.1:{vllm_wrapper_port}");
        let mut last_error = String::new();

        for attempt in 1..=3 {
            let mut client = match proto::vllm_wrapper_client::VllmWrapperClient::connect(
                endpoint.clone(),
            )
            .await
            {
                Ok(client) => client,
                Err(error) => {
                    last_error = format!("connect error: {error}");
                    if attempt == 3 {
                        break;
                    }
                    continue;
                }
            };

            let proto_request = request.to_proto();
            let rpc_result = client.generate(Request::new(proto_request)).await;
            match rpc_result {
                Ok(response) => {
                    return VllmResponse::from_proto(response.into_inner());
                }
                Err(status) => {
                    last_error = format!("grpc status {:?}: {}", status.code(), status);
                    if attempt == 3 {
                        break;
                    }
                }
            }
        }

        VllmResponse::Error {
            error_message: format!(
                "vLLM-wrapper gRPC call failed after 3 retries on {}: {}",
                endpoint, last_error
            ),
        }
    }
}

fn is_context_length_error(error_message: &str) -> bool {
    error_message.contains("maximum context length")
        || error_message.contains("Please reduce the length of the input prompt")
        || error_message.contains("parameter=input_tokens")
        || error_message.contains("context length")
}

fn json_compact(value: &Value) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|err| format!("{{\"json_serialize_error\":\"{}\"}}", err))
}

fn parse_sglang_generated_token_ids(
    json: &Value,
    input_tokens: &[i32],
    backend_label: &str,
) -> Vec<i32> {
    let output_ids = json["output_ids"].as_array().unwrap_or_else(|| {
        panic!(
            "Qwen SGLang response missing output_ids on {}: {}",
            backend_label,
            json_compact(json)
        )
    });

    let all_ids: Vec<i32> = output_ids
        .iter()
        .map(|token| {
            let raw = token.as_i64().unwrap_or_else(|| {
                panic!(
                    "Qwen SGLang output_ids entry must be an integer on {}: {:?}",
                    backend_label, token
                )
            });
            i32::try_from(raw).unwrap_or_else(|_| {
                panic!(
                    "Qwen SGLang token id must fit in i32 on {}: {:?}",
                    backend_label, token
                )
            })
        })
        .collect();

    if all_ids.starts_with(input_tokens) {
        return all_ids[input_tokens.len()..].to_vec();
    }

    all_ids
}

fn parse_sglang_response_with_logprobs(
    backend_label: &str,
    json: &Value,
    input_tokens: &[i32],
    decode_tokens: fn(&[i32]) -> String,
) -> TokenArrayWithLogprob {
    if let Some(error_message) = json["error"]["message"].as_str() {
        panic!(
            "Qwen completion with logprobs failed on {}: {}. Full response: {}",
            backend_label,
            error_message,
            json_compact(json)
        );
    }

    let generated_tokens = parse_sglang_generated_token_ids(json, input_tokens, backend_label);

    let token_logprobs = json["meta_info"]["output_token_logprobs"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "Qwen SGLang response missing meta_info.output_token_logprobs on {}: {}",
                backend_label,
                json_compact(json)
            )
        });

    let top_logprobs = json["meta_info"]["output_top_logprobs"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "Qwen SGLang response missing meta_info.output_top_logprobs on {}: {}",
                backend_label,
                json_compact(json)
            )
        });

    let token_logprob_token_ids =
        parse_sglang_output_token_ids_from_token_logprobs(token_logprobs, backend_label);
    assert!(
        token_logprobs.len() == generated_tokens.len(),
        "Qwen SGLang output_token_logprobs length mismatch on {}: output_token_logprobs={} generated_tokens={} generated_token_ids={:?} output_token_logprob_token_ids={:?}",
        backend_label,
        token_logprobs.len(),
        generated_tokens.len(),
        generated_tokens,
        token_logprob_token_ids,
    );
    assert!(
        top_logprobs.len() == generated_tokens.len(),
        "Qwen SGLang output_top_logprobs length mismatch on {}: output_top_logprobs={} generated_tokens={} generated_token_ids={:?}",
        backend_label,
        top_logprobs.len(),
        generated_tokens.len(),
        generated_tokens,
    );

    let decoded_string = decode_tokens(&generated_tokens);

    let mut aligned_logprobs = Vec::with_capacity(generated_tokens.len());
    for (idx, generated_token_id) in generated_tokens.iter().copied().enumerate() {
        let generated_logprob = token_logprobs[idx][0]
            .as_f64()
            .map(|value| value as f32)
            .unwrap_or(f32::NEG_INFINITY);

        let mut candidates = parse_sglang_top_logprob_candidates(
            &top_logprobs[idx],
            backend_label,
            generated_token_id,
        );

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

fn parse_sglang_output_token_ids_from_token_logprobs(
    token_logprobs: &[Value],
    backend_label: &str,
) -> Vec<i32> {
    token_logprobs
        .iter()
        .map(|entry| {
            let entry_items = entry.as_array().unwrap_or_else(|| {
                panic!(
                    "Qwen SGLang output_token_logprob entry must be an array on {}: {:?}",
                    backend_label, entry
                )
            });
            assert!(
                entry_items.len() >= 2,
                "Qwen SGLang output_token_logprob entry must have at least 2 fields on {}: {:?}",
                backend_label,
                entry
            );
            let token_id_raw = entry_items[1].as_i64().unwrap_or_else(|| {
                panic!(
                    "Qwen SGLang output_token_logprob token id must be an integer on {}: {:?}",
                    backend_label, entry
                )
            });
            i32::try_from(token_id_raw).unwrap_or_else(|_| {
                panic!(
                    "Qwen SGLang output_token_logprob token id must fit in i32 on {}: {:?}",
                    backend_label, entry
                )
            })
        })
        .collect()
}

fn parse_sglang_top_logprob_candidates(
    value: &Value,
    backend_label: &str,
    fallback_token_id: i32,
) -> Vec<TokenLogprobCandidate> {
    let Some(candidates) = value.as_array() else {
        return vec![TokenLogprobCandidate {
            token_id: fallback_token_id,
            logprob: f32::NEG_INFINITY,
        }];
    };

    candidates
        .iter()
        .filter_map(|entry| {
            let entry_items = entry.as_array().unwrap_or_else(|| {
                panic!(
                    "Qwen SGLang top_logprob entry must be an array on {}: {:?}",
                    backend_label, entry
                )
            });
            assert!(
                entry_items.len() >= 2,
                "Qwen SGLang top_logprob entry must have at least 2 fields on {}: {:?}",
                backend_label,
                entry
            );

            let logprob = entry_items[0].as_f64().unwrap_or_else(|| {
                panic!(
                    "Qwen SGLang top_logprob value must be f64-compatible on {}: {:?}",
                    backend_label, entry
                )
            }) as f32;

            let token_id_raw = entry_items[1].as_i64().unwrap_or_else(|| {
                panic!(
                    "Qwen SGLang top_logprob token id must be an integer on {}: {:?}",
                    backend_label, entry
                )
            });

            i32::try_from(token_id_raw)
                .ok()
                .map(|token_id| TokenLogprobCandidate { token_id, logprob })
        })
        .collect()
}

fn parse_completion_response_with_logprobs(
    backend_label: &str,
    json: &Value,
    encode_text: fn(&str) -> Vec<i32>,
) -> TokenArrayWithLogprob {
    if let Some(error_message) = json["error"]["message"].as_str() {
        panic!(
            "Qwen completion with logprobs failed on {}: {}. Full response: {:?}",
            backend_label, error_message, json
        );
    }

    let choice = &json["choices"][0];
    let decoded_string = choice["text"]
        .as_str()
        .unwrap_or_else(|| {
            panic!(
                "Qwen completion response missing choices[0].text on {}: {:?}",
                backend_label, json
            )
        })
        .to_string();

    let token_strings = choice["logprobs"]["tokens"].as_array().unwrap_or_else(|| {
        panic!(
            "Qwen completion response missing choices[0].logprobs.tokens on {}: {:?}",
            backend_label, json
        )
    });

    let generated_tokens: Vec<i32> = token_strings
        .iter()
        .map(|token| {
            let token = token.as_str().unwrap_or_else(|| {
                panic!(
                    "Qwen completion logprobs.tokens entry must be a string on {}: {:?}",
                    backend_label, token
                )
            });
            let encoded = encode_text(token);
            assert!(
                encoded.len() == 1,
                "Qwen completion logprobs.tokens entry must map to exactly one token id on {}: token={:?} encoded={:?}",
                backend_label,
                token,
                encoded
            );
            encoded[0]
        })
        .collect();

    let token_logprobs = choice["logprobs"]["token_logprobs"].as_array();
    let top_logprobs = choice["logprobs"]["top_logprobs"].as_array();

    if let Some(values) = token_logprobs {
        assert!(
            values.len() == generated_tokens.len(),
            "Qwen completion token_logprobs length mismatch on {}: token_logprobs={} generated_tokens={}. Full response: {:?}",
            backend_label,
            values.len(),
            generated_tokens.len(),
            json
        );
    }
    if let Some(values) = top_logprobs {
        assert!(
            values.len() == generated_tokens.len(),
            "Qwen completion top_logprobs length mismatch on {}: top_logprobs={} generated_tokens={}. Full response: {:?}",
            backend_label,
            values.len(),
            generated_tokens.len(),
            json
        );
    }

    let mut aligned_logprobs = Vec::with_capacity(generated_tokens.len());
    for (idx, generated_token_id) in generated_tokens.iter().copied().enumerate() {
        let generated_logprob = token_logprobs
            .and_then(|vals| vals.get(idx))
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(f32::NEG_INFINITY);

        let mut candidates = top_logprobs
            .and_then(|vals| vals.get(idx))
            .map(|value| parse_completion_candidates(value, backend_label, encode_text))
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
                        let logprob =
                            candidate["logprob"].as_f64().unwrap_or(f64::NEG_INFINITY) as f32;
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

fn parse_completion_candidates(
    value: &Value,
    backend_label: &str,
    encode_text: fn(&str) -> Vec<i32>,
) -> Vec<TokenLogprobCandidate> {
    if let Some(map) = value.as_object() {
        return map
            .iter()
            .map(|(token, logprob)| {
                let logprob = logprob.as_f64().unwrap_or_else(|| {
                    panic!(
                        "Completion top_logprobs value must be f64-compatible on {}: token={:?} value={:?}",
                        backend_label, token, logprob
                    )
                }) as f32;
                let encoded = encode_text(token);
                assert!(
                    encoded.len() == 1,
                    "Completion top_logprobs token must map to exactly one token id on {}: token={:?} encoded={:?}, json object: {:?}",
                    backend_label,
                    token,
                    encoded,
                    value
                );
                TokenLogprobCandidate {
                    token_id: encoded[0],
                    logprob,
                }
            })
            .collect();
    }

    panic!(
        "Unexpected completion top_logprobs shape on {}; expected object mapping token string -> logprob. Got: {:?}",
        backend_label, value
    )
}
