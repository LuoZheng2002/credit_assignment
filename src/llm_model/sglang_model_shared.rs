use reqwest::Client;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::constants::sglang_context_length;
use crate::llm_model::LlmModelMarker;
use crate::token_array::{TokenArrayWithLogprob, TokenLogprobCandidate};

pub(crate) fn wrap_python_response_xml(raw_python_response: &str) -> String {
    format!("<tool_response>{}</tool_response>", raw_python_response)
}

fn build_qwen_python_response_turn(
    raw_python_response: &str,
    assistant_response_start: &str,
) -> String {
    let wrapped_response = wrap_python_response_xml(raw_python_response);
    format!(
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n{}",
        wrapped_response, assistant_response_start
    )
}

pub(crate) fn build_qwen25_python_response_turn_enable_thinking(
    raw_python_response: &str,
) -> String {
    build_qwen_python_response_turn(raw_python_response, "<|im_start|>assistant\n")
}

pub(crate) fn build_qwen25_python_response_turn_disable_thinking(
    raw_python_response: &str,
) -> String {
    build_qwen_python_response_turn(raw_python_response, "<|im_start|>assistant\n")
}

pub(crate) fn build_qwen3_python_response_turn_enable_thinking(
    raw_python_response: &str,
) -> String {
    build_qwen_python_response_turn(raw_python_response, "<|im_start|>assistant\n")
}

pub(crate) fn build_qwen3_python_response_turn_disable_thinking(
    raw_python_response: &str,
) -> String {
    build_qwen_python_response_turn(
        raw_python_response,
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
}

pub(crate) fn build_qwen35_python_response_turn_enable_thinking(
    raw_python_response: &str,
) -> String {
    build_qwen_python_response_turn(raw_python_response, "<|im_start|>assistant\n<think>\n")
}

pub(crate) fn build_qwen35_python_response_turn_disable_thinking(
    raw_python_response: &str,
) -> String {
    build_qwen_python_response_turn(
        raw_python_response,
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
}

fn remaining_generation_tokens(prompt_len: usize, use_tool: bool) -> Result<usize, String> {
    let context_length = sglang_context_length(use_tool);
    let remaining = context_length.checked_sub(prompt_len + 1).ok_or_else(|| {
        format!(
            "Context length exceeded before generation (prompt_length={}, limit={}).",
            prompt_len, context_length
        )
    })?;
    if remaining == 0 {
        return Err(format!(
            "Context length exceeded before generation (prompt_length={}, limit={}).",
            prompt_len, context_length
        ));
    }
    Ok(remaining)
}

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
pub(crate) struct SharedSglangLlmCallable {
    client: Client,
    sglang_port: u16,
}

impl SharedSglangLlmCallable {
    pub(crate) fn new(client: Client, sglang_port: u16) -> Self {
        assert!(sglang_port > 0, "SGLang port must be greater than 0");
        Self {
            client,
            sglang_port,
        }
    }

    pub(crate) async fn generate_tokens_from_tokens<M: LlmModelMarker>(
        &self,
        tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> Result<Vec<i32>, String> {
        let max_new_tokens = remaining_generation_tokens(tokens.len(), passes_in_stop)?;
        let mut body = serde_json::json!({
            "input_ids": tokens.clone(),
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "no_stop_trim": true,
            },
            "stream": false,
        });
        if passes_in_stop {
            body["sampling_params"]["stop"] = serde_json::json!(["```\n"]);
        }

        let response = self
            .post_json(
                &format!("http://localhost:{}/generate", self.sglang_port),
                body,
            )
            .await?;

        if let Some(error_message) = response["error"]["message"].as_str() {
            return Err(error_message.to_string());
        }

        let generated_tokens = parse_sglang_generated_token_ids(
            &response,
            &tokens,
            &format!("SGLang port {}", self.sglang_port),
        )?;
        Ok(generated_tokens)
    }

    pub(crate) async fn generate_tokens_with_logprobs_from_tokens<M: LlmModelMarker>(
        &self,
        tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
    ) -> Result<TokenArrayWithLogprob<M>, String> {
        let max_new_tokens = remaining_generation_tokens(tokens.len(), passes_in_stop)?;
        let logprob_start_len: i64 = if M::CLI_NAME.starts_with("gemma-") {
            tokens.len() as i64
        } else {
            -1
        };
        let mut body = serde_json::json!({
            "input_ids": tokens.clone(),
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "no_stop_trim": true,
            },
            "return_logprob": true,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 8,
            "stream": false,
        });
        if passes_in_stop {
            body["sampling_params"]["stop"] = serde_json::json!(["```\n"]);
        }

        let json = self
            .post_json(
                &format!("http://localhost:{}/generate", self.sglang_port),
                body,
            )
            .await?;
        let result = parse_sglang_response_with_logprobs(
            &format!("SGLang port {}", self.sglang_port),
            &json,
        )?;
        Ok(result)
    }

    async fn post_json(&self, url: &str, body: serde_json::Value) -> Result<Value, String> {
        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|err| format!("SGLang request failed to {}: {}", url, err))?;
        response
            .json()
            .await
            .map_err(|err| format!("SGLang failed to parse JSON from {}: {}", url, err))
    }
}

fn json_compact(value: &Value) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|err| format!("{{\"json_serialize_error\":\"{}\"}}", err))
}

fn parse_sglang_generated_token_ids(
    json: &Value,
    input_tokens: &[i32],
    backend_label: &str,
) -> Result<Vec<i32>, String> {
    let output_ids = json["output_ids"].as_array().ok_or_else(|| {
        format!(
            "SGLang response missing output_ids on {}: {}",
            backend_label,
            json_compact(json)
        )
    })?;

    let all_ids: Vec<i32> = output_ids
        .iter()
        .map(|token| {
            let raw = token.as_i64().ok_or_else(|| {
                format!(
                    "SGLang output_ids entry must be an integer on {}: {:?}",
                    backend_label, token
                )
            })?;
            i32::try_from(raw).map_err(|_| {
                format!(
                    "SGLang token id must fit in i32 on {}: {:?}",
                    backend_label, token
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if all_ids.starts_with(input_tokens) {
        return Ok(all_ids[input_tokens.len()..].to_vec());
    }

    Ok(all_ids)
}

fn parse_sglang_response_with_logprobs<M>(
    backend_label: &str,
    json: &Value,
) -> Result<TokenArrayWithLogprob<M>, String> {
    if let Some(error_message) = json["error"]["message"].as_str() {
        return Err(format!(
            "SGLang completion with logprobs failed on {}: {}. Full response: {}",
            backend_label,
            error_message,
            json_compact(json)
        ));
    }

    let token_logprobs = json["meta_info"]["output_token_logprobs"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "SGLang response missing meta_info.output_token_logprobs on {}: {}",
                backend_label,
                json_compact(json)
            )
        });

    let top_logprobs = json["meta_info"]["output_top_logprobs"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "SGLang response missing meta_info.output_top_logprobs on {}: {}",
                backend_label,
                json_compact(json)
            )
        });

    let generated_tokens =
        parse_sglang_output_token_ids_from_token_logprobs(token_logprobs, backend_label);
    assert!(
        top_logprobs.len() == generated_tokens.len(),
        "SGLang output_top_logprobs length mismatch on {}: output_top_logprobs={} generated_tokens={} generated_token_ids={:?}",
        backend_label,
        top_logprobs.len(),
        generated_tokens.len(),
        generated_tokens,
    );

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

    Ok(TokenArrayWithLogprob::from_tokens_and_logprobs(
        generated_tokens,
        aligned_logprobs,
    ))
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
                    "SGLang output_token_logprob entry must be an array on {}: {:?}",
                    backend_label, entry
                )
            });
            assert!(
                entry_items.len() >= 2,
                "SGLang output_token_logprob entry must have at least 2 fields on {}: {:?}",
                backend_label,
                entry
            );
            let token_id_raw = entry_items[1].as_i64().unwrap_or_else(|| {
                panic!(
                    "SGLang output_token_logprob token id must be an integer on {}: {:?}",
                    backend_label, entry
                )
            });
            i32::try_from(token_id_raw).unwrap_or_else(|_| {
                panic!(
                    "SGLang output_token_logprob token id must fit in i32 on {}: {:?}",
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
                    "SGLang top_logprob entry must be an array on {}: {:?}",
                    backend_label, entry
                )
            });
            assert!(
                entry_items.len() >= 2,
                "SGLang top_logprob entry must have at least 2 fields on {}: {:?}",
                backend_label,
                entry
            );

            let logprob = entry_items[0].as_f64().unwrap_or_else(|| {
                panic!(
                    "SGLang top_logprob value must be f64-compatible on {}: {:?}",
                    backend_label, entry
                )
            }) as f32;

            let token_id_raw = entry_items[1].as_i64().unwrap_or_else(|| {
                panic!(
                    "SGLang top_logprob token id must be an integer on {}: {:?}",
                    backend_label, entry
                )
            });

            i32::try_from(token_id_raw)
                .ok()
                .map(|token_id| TokenLogprobCandidate { token_id, logprob })
        })
        .collect()
}
