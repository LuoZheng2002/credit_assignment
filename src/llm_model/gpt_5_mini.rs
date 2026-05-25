use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use std::sync::LazyLock;
use std::time::Duration;
use tiktoken_rs::{CoreBPE, bpe_for_model};

use super::{
    LlmCallable, LlmCliArgs, LlmFamily, LlmModelMarker, MyTokenizer, TokenArray,
    TokenArrayWithLogprob, TokenLogprobCandidate,
};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

static GPT5_MINI_TOKENIZER: LazyLock<CoreBPE> =
    LazyLock::new(|| bpe_for_model(Gpt5Mini::API_NAME).unwrap().clone());

#[derive(Clone)]
pub struct Gpt5MiniLlmCallable {
    client: Client,
}

impl Gpt5MiniLlmCallable {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    async fn post_json(&self, url: &str, body: serde_json::Value) -> reqwest::Response {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
        const MAX_ATTEMPTS: usize = 3;

        for attempt in 1..=MAX_ATTEMPTS {
            match self
                .client
                .post(url)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(response) => return response,
                Err(err)
                    if attempt < MAX_ATTEMPTS
                        && (err.is_connect() || err.is_timeout() || err.is_request()) =>
                {
                    let backoff_ms = 250_u64 * (attempt as u64);
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
                Err(err) => {
                    panic!(
                        "OpenAI request failed (attempt {attempt}/{MAX_ATTEMPTS}) to {url}: {err:?}"
                    )
                }
            }
        }

        panic!("OpenAI request failed unexpectedly without returning an error")
    }
}

#[async_trait]
impl LlmCallable<Gpt5Mini> for Gpt5MiniLlmCallable {
    async fn generate_text(&self, prompt_or_tokens: Vec<i32>, passes_in_stop: bool) -> String {
        let prompt = <Gpt5Mini as LlmModelMarker>::Tokenizer::decode_i32_ids(&prompt_or_tokens);
        let body = if passes_in_stop {
            serde_json::json!({
                "model": Gpt5Mini::API_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2048,
                "stop": ["</tool_wait>"],
            })
        } else {
            serde_json::json!({
                "model": Gpt5Mini::API_NAME,
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
            .unwrap_or_else(|| panic!("LLM response is invalid: {:?}", json))
            .to_string()
    }

    async fn generate_tokens_with_logprobs(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
    ) -> TokenArrayWithLogprob {
        let prompt = <Gpt5Mini as LlmModelMarker>::Tokenizer::decode_i32_ids(&prompt_or_tokens);
        let body = if passes_in_stop {
            serde_json::json!({
                "model": Gpt5Mini::API_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2048,
                "stop": ["</tool_wait>"],
                "temperature": temperature,
                "logprobs": true,
                "top_logprobs": 8,
            })
        } else {
            serde_json::json!({
                "model": Gpt5Mini::API_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2048,
                "temperature": temperature,
                "logprobs": true,
                "top_logprobs": 8,
            })
        };

        let response = self
            .post_json(OPENAI_CHAT_COMPLETIONS_URL, body)
            .await
            .bytes()
            .await
            .unwrap();
        let Ok(json) = serde_json::from_slice::<Value>(&response) else {
            panic!(
                "Failed to parse LLM response as JSON. Response text: {:?}",
                String::from_utf8_lossy(&response)
            );
        };

        let content_entries = json["choices"][0]["logprobs"]["content"]
            .as_array()
            .unwrap_or_else(|| {
                panic!("LLM response is invalid (missing logprobs.content): {json:?}")
            });

        let mut tokens = Vec::with_capacity(content_entries.len());
        let mut decoded_string = String::new();
        let mut logprobs = Vec::with_capacity(content_entries.len());

        for entry in content_entries {
            let sampled_token = entry["token"]
                .as_str()
                .unwrap_or_else(|| panic!("LLM logprob entry missing token: {entry:?}"));
            let Some(sampled_token_id) = try_token_to_single_id::<Gpt5Mini>(sampled_token) else {
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
                            let token_id = try_token_to_single_id::<Gpt5Mini>(token)?;
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

        TokenArrayWithLogprob {
            tokens,
            decoded_string,
            logprobs,
        }
    }
}

fn try_token_to_single_id<M: LlmModelMarker>(token: &str) -> Option<i32> {
    let encoded = <M as LlmModelMarker>::Tokenizer::encode_to_i32_ids(token);
    if encoded.len() == 1 {
        Some(encoded[0])
    } else {
        None
    }
}

pub struct Gpt5Mini;

pub struct Gpt5MiniTokenizer;
impl MyTokenizer<Gpt5Mini> for Gpt5MiniTokenizer {
    fn tokenize(prompt: String) -> TokenArray {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        GPT5_MINI_TOKENIZER
            .encode_with_special_tokens(text)
            .into_iter()
            .map(|token| i32::try_from(token).expect("token id must fit in i32"))
            .collect()
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        let token_ids: Vec<u32> = token_ids
            .iter()
            .map(|token| u32::try_from(*token).expect("token id must be non-negative"))
            .collect();
        GPT5_MINI_TOKENIZER
            .decode(&token_ids)
            .expect("failed to decode GPT token ids")
    }

    fn token_to_id(token: &str) -> i32 {
        let encoded = GPT5_MINI_TOKENIZER.encode_with_special_tokens(token);
        assert_eq!(
            encoded.len(),
            1,
            "Token '{token}' must map to a single GPT token for model {}",
            Gpt5Mini::API_NAME
        );
        i32::try_from(encoded[0]).expect("token id must fit in i32")
    }
}

impl LlmModelMarker for Gpt5Mini {
    type Tokenizer = Gpt5MiniTokenizer;
    type Callable = Gpt5MiniLlmCallable;

    const CLI_NAME: &'static str = "gpt-5-mini";
    const API_NAME: &'static str = "gpt-5-mini";
    const FAMILY: LlmFamily = LlmFamily::Gpt;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        format!(
            "{}\nAssistant: {}",
            prompt_before_assistant, prompt_after_assistant
        )
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        prompt_before_assistant.to_string()
    }

    fn callable_from_cli_args(client: Client, _llm_cli_args: &LlmCliArgs) -> Self::Callable {
        Gpt5MiniLlmCallable::new(client)
    }
}
