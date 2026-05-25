use async_trait::async_trait;
use reqwest::Client;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use super::qwen_shared::{
    QwenBackend, SharedQwenLlmCallable, decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id,
};
use super::{
    LlmCallable, LlmCliArgs, LlmFamily, LlmModelMarker, MyTokenizer, QwenApiBackend, TokenArray,
    TokenArrayWithLogprob, build_simple_qwen_chatml_prefix,
};

static QWEN35_4B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen35_4B::API_NAME, None).unwrap());

pub struct Qwen35_4B;

#[derive(Clone)]
pub struct Qwen35_4BLlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen35_4BLlmCallable {
    pub(crate) fn new(
        client: Client,
        backend: QwenBackend,
        max_concurrent_requests: usize,
    ) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(
                client,
                Qwen35_4B::API_NAME,
                backend,
                max_concurrent_requests,
                decode_qwen35_4b_tokens,
                encode_qwen35_4b_text,
            ),
        }
    }
}

fn decode_qwen35_4b_tokens(token_ids: &[i32]) -> String {
    decode_from_i32_ids(&QWEN35_4B_TOKENIZER, token_ids)
}

fn encode_qwen35_4b_text(text: &str) -> Vec<i32> {
    encode_to_i32_ids(&QWEN35_4B_TOKENIZER, text)
}

#[async_trait]
impl LlmCallable<Qwen35_4B> for Qwen35_4BLlmCallable {
    async fn generate_text(&self, prompt_or_tokens: Vec<i32>, passes_in_stop: bool) -> String {
        self.shared
            .generate_from_tokens(prompt_or_tokens, passes_in_stop)
            .await
    }

    async fn generate_tokens_with_logprobs(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
    ) -> TokenArrayWithLogprob {
        self.shared
            .generate_tokens_with_logprobs_from_tokens(prompt_or_tokens, passes_in_stop, temperature)
            .await
    }
}

pub struct Qwen35_4BTokenizer;
impl MyTokenizer<Qwen35_4B> for Qwen35_4BTokenizer {
    fn tokenize(prompt: String) -> TokenArray {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&QWEN35_4B_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&QWEN35_4B_TOKENIZER, token_ids)
    }

    fn token_to_id(token: &str) -> i32 {
        token_to_i32_id(&QWEN35_4B_TOKENIZER, token, Qwen35_4B::API_NAME)
    }
}

impl LlmModelMarker for Qwen35_4B {
    type Tokenizer = Qwen35_4BTokenizer;
    type Callable = Qwen35_4BLlmCallable;

    const CLI_NAME: &'static str = "qwen3.5-4b";
    const API_NAME: &'static str = "Qwen/Qwen3.5-4B";
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
        let backend = match llm_cli_args.qwen_api_backend {
            QwenApiBackend::Vllm => QwenBackend::Vllm {
                vllm_port: llm_cli_args.qwen_vllm_port(),
            },
            QwenApiBackend::Sglang => QwenBackend::Sglang {
                sglang_port: llm_cli_args.qwen_sglang_port(),
            },
            QwenApiBackend::VllmWrapper => QwenBackend::VllmWrapper {
                vllm_wrapper_port: llm_cli_args.vllm_wrapper_port(),
            },
            QwenApiBackend::Openrouter => QwenBackend::OpenRouter {
                base_url: llm_cli_args.openrouter_base_url.clone(),
                model: llm_cli_args.openrouter_model_or_default(Self::API_NAME),
                api_key: llm_cli_args.openrouter_api_key(),
                http_referer: llm_cli_args.openrouter_http_referer.clone(),
                x_title: llm_cli_args.openrouter_x_title.clone(),
            },
        };

        Qwen35_4BLlmCallable::new(client, backend, llm_cli_args.max_concurrent_requests)
    }
}
