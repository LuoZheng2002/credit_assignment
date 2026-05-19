use async_trait::async_trait;
use reqwest::Client;
use std::sync::LazyLock;
use tiktoken_rs::{CoreBPE, bpe_for_model};

use crate::token_array::TokenArray;

use super::{LlmCallable, LlmCliArgs, LlmFamily, LlmModelMarker, MyTokenizer};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

static GPT4O_TOKENIZER: LazyLock<CoreBPE> =
    LazyLock::new(|| bpe_for_model(Gpt4o::API_NAME).unwrap().clone());

#[derive(Clone)]
pub struct Gpt4oLlmCallable {
    client: Client,
}

impl Gpt4oLlmCallable {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

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

#[async_trait]
impl LlmCallable<Gpt4o> for Gpt4oLlmCallable {
    async fn generate(&self, prompt_or_tokens: Vec<i32>, passes_in_stop: bool) -> String {
        let prompt = <Gpt4o as LlmModelMarker>::Tokenizer::decode_i32_ids(&prompt_or_tokens);
        let body = if passes_in_stop {
            serde_json::json!({
                "model": Gpt4o::API_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2048,
                "stop": ["</tool_wait>"],
            })
        } else {
            serde_json::json!({
                "model": Gpt4o::API_NAME,
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
}

pub struct Gpt4o;

pub struct Gpt4oTokenizer;
impl MyTokenizer<Gpt4o> for Gpt4oTokenizer {
    fn tokenize(prompt: String) -> TokenArray {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        GPT4O_TOKENIZER
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
        GPT4O_TOKENIZER
            .decode(&token_ids)
            .expect("failed to decode GPT token ids")
    }

    fn token_to_id(token: &str) -> i32 {
        let encoded = GPT4O_TOKENIZER.encode_with_special_tokens(token);
        assert_eq!(
            encoded.len(),
            1,
            "Token '{token}' must map to a single GPT token for model {}",
            Gpt4o::API_NAME
        );
        i32::try_from(encoded[0]).expect("token id must fit in i32")
    }
}

impl LlmModelMarker for Gpt4o {
    type Tokenizer = Gpt4oTokenizer;
    type Callable = Gpt4oLlmCallable;

    const CLI_NAME: &'static str = "gpt-4o";
    const API_NAME: &'static str = "gpt-4o";
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
        Gpt4oLlmCallable::new(client)
    }
}
