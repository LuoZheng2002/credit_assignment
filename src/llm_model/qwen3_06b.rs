use async_trait::async_trait;
use reqwest::Client;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use super::qwen_shared::{
    SharedQwenLlmCallable, build_qwen3_python_response_turn_disable_thinking,
    build_qwen3_python_response_turn_enable_thinking, decode_from_i32_ids, encode_to_i32_ids,
    token_to_i32_id,
};
use super::{
    LlmCallable, LlmCliArgs, LlmModelMarker, MyTokenizer, TokenArray, TokenArrayWithLogprob,
    build_simple_qwen3_chatml_template, trim_tail_eos_if_needed,
};

static QWEN3_06B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen3_06B::API_NAME, None).unwrap());

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Qwen3_06B;

#[derive(Clone)]
pub struct Qwen3_06BLlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen3_06BLlmCallable {
    pub(crate) fn new(client: Client, sglang_port: u16) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(client, sglang_port),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen3_06B> for Qwen3_06BLlmCallable {
    fn from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self {
        let sglang_port = llm_cli_args
            .sglang_port
            .expect("Qwen3-0.6B model requires sglang port");

        Qwen3_06BLlmCallable::new(client, sglang_port)
    }

    async fn generate_tokens(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> Result<Vec<i32>, String> {
        self.shared
            .generate_tokens_from_tokens(prompt_or_tokens, passes_in_stop)
            .await
    }

    async fn generate_tokens_with_logprobs(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
        trim_eos: bool,
    ) -> Result<TokenArrayWithLogprob<Qwen3_06B>, String> {
        let output = self
            .shared
            .generate_tokens_with_logprobs_from_tokens(
                prompt_or_tokens,
                passes_in_stop,
                temperature,
            )
            .await?;
        Ok(trim_tail_eos_if_needed::<Qwen3_06B>(output, trim_eos))
    }
}

pub struct Qwen3_06BTokenizer;
impl MyTokenizer<Qwen3_06B> for Qwen3_06BTokenizer {
    fn tokenize(prompt: String) -> TokenArray<Qwen3_06B> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn apply_chat_template_and_tokenize(
        prompt: String,
        enable_thinking: bool,
    ) -> TokenArray<Qwen3_06B> {
        let prompt_with_template = build_simple_qwen3_chatml_template(&prompt, enable_thinking);
        Self::tokenize(prompt_with_template)
    }

    fn apply_python_response_template_and_tokenize(
        raw_python_response: String,
        enable_thinking: bool,
    ) -> TokenArray<Qwen3_06B> {
        let wrapped_turn = if enable_thinking {
            build_qwen3_python_response_turn_enable_thinking(&raw_python_response)
        } else {
            build_qwen3_python_response_turn_disable_thinking(&raw_python_response)
        };
        Self::tokenize(wrapped_turn)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&QWEN3_06B_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&QWEN3_06B_TOKENIZER, token_ids)
    }

    fn eos_token_id() -> i32 {
        token_to_i32_id(&QWEN3_06B_TOKENIZER, "<|im_end|>", Qwen3_06B::API_NAME)
    }
}

impl LlmModelMarker for Qwen3_06B {
    type Tokenizer = Qwen3_06BTokenizer;
    type Callable = Qwen3_06BLlmCallable;

    const CLI_NAME: &'static str = "qwen3-0.6b";
    const API_NAME: &'static str = "Qwen/Qwen3-0.6B";
}
