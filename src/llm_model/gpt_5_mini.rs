use async_trait::async_trait;
use reqwest::Client;

use super::{LlmCallable, LlmCliArgs, LlmFamily, LlmModelMarker, PassthroughTokenizer};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

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
impl LlmCallable<Gpt5Mini> for Gpt5MiniLlmCallable {
    async fn generate(&self, prompt: String, passes_in_stop: bool) -> String {
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
}

pub struct Gpt5Mini;
impl LlmModelMarker for Gpt5Mini {
    type StringOrTokenArray = String;
    type Tokenizer = PassthroughTokenizer;
    type Callable = Gpt5MiniLlmCallable;

    const CLI_NAME: &'static str = "gpt-5-mini";
    const API_NAME: &'static str = "gpt-5-mini";
    const FAMILY: LlmFamily = LlmFamily::Gpt;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        format!("{}\nAssistant: {}", prompt_before_assistant, prompt_after_assistant)
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        prompt_before_assistant.to_string()
    }

    fn callable_from_cli_args(client: Client, _llm_cli_args: &LlmCliArgs) -> Self::Callable {
        Gpt5MiniLlmCallable::new(client)
    }
}
