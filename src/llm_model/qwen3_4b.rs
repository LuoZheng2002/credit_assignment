use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use super::{
    LlmCallable, LlmCliArgs, LlmFamily, LlmModelMarker, MyTokenizer,
    build_simple_qwen_chatml_prefix,
};
use super::qwen_shared::{
    SharedQwenLlmCallable, encode_to_i32_ids, token_to_i32_id,
};

static QWEN3_4B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen3_4B::API_NAME, None).unwrap());

pub struct Qwen3_4B;

#[derive(Clone)]
pub struct Qwen3_4BLlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen3_4BLlmCallable {
    pub fn new(client: Client, vllm_port: u16, max_concurrent_requests: usize) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(
                client,
                Qwen3_4B::API_NAME,
                vllm_port,
                max_concurrent_requests,
            ),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen3_4B> for Qwen3_4BLlmCallable {
    async fn generate(&self, prompt_or_tokens: Qwen3TokenArray, passes_in_stop: bool) -> String {
        self.shared
            .generate_from_tokens(prompt_or_tokens.tokens, passes_in_stop)
            .await
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Qwen3TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}

pub struct Qwen3_4BTokenizer;
impl MyTokenizer<Qwen3_4B> for Qwen3_4BTokenizer {
    fn tokenize(prompt: String) -> Qwen3TokenArray {
        let tokens = Self::encode_to_i32_ids(&prompt);
        Qwen3TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&QWEN3_4B_TOKENIZER, text)
    }

    fn token_to_id(token: &str) -> i32 {
        token_to_i32_id(&QWEN3_4B_TOKENIZER, token, Qwen3_4B::API_NAME)
    }
}

impl LlmModelMarker for Qwen3_4B {
    type StringOrTokenArray = Qwen3TokenArray;
    type Tokenizer = Qwen3_4BTokenizer;
    type Callable = Qwen3_4BLlmCallable;

    const CLI_NAME: &'static str = "qwen3-4b";
    const API_NAME: &'static str = "Qwen/Qwen3-4B";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt = build_simple_qwen_chatml_prefix(prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        build_simple_qwen_chatml_prefix(prompt_before_assistant, true)
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable {
        Qwen3_4BLlmCallable::new(
            client,
            llm_cli_args.single_port_for_qwen(),
            llm_cli_args.max_concurrent_requests,
        )
    }
}
