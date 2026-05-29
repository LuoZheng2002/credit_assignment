use async_trait::async_trait;
use reqwest::Client;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::{
    llm_model::{
        TokenArrayWithLogprob,
        llm_model_traits::{
            LlmCallable, LlmCliArgs, LlmModelMarker, MyTokenizer, trim_tail_eos_if_needed,
        },
    },
    token_array::TokenArray,
};

use super::qwen_shared::{
    SharedQwenLlmCallable, decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id,
};

static QWEN3_4B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen3_4B::API_NAME, None).unwrap());

pub struct Qwen3_4B;

#[derive(Clone)]
pub struct Qwen3_4BLlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen3_4BLlmCallable {
    pub(crate) fn new(client: Client, sglang_port: u16) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(client, sglang_port),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen3_4B> for Qwen3_4BLlmCallable {
    fn from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self {
        let sglang_port = llm_cli_args
            .sglang_port
            .expect("Qwen3-4B model requires sglang port");

        Qwen3_4BLlmCallable::new(client, sglang_port)
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
    ) -> Result<TokenArrayWithLogprob<Qwen3_4B>, String> {
        let output = self
            .shared
            .generate_tokens_with_logprobs_from_tokens(
                prompt_or_tokens,
                passes_in_stop,
                temperature,
            )
            .await?;
        Ok(trim_tail_eos_if_needed::<Qwen3_4B>(output, trim_eos))
    }
}

pub(crate) fn build_simple_qwen3_chatml_template(
    user_prompt: &str,
    enable_thinking: bool,
) -> String {
    if enable_thinking {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_prompt
        )
    } else {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            user_prompt
        )
    }
}

pub struct Qwen3_4BTokenizer;
impl MyTokenizer<Qwen3_4B> for Qwen3_4BTokenizer {
    fn tokenize(prompt: String) -> TokenArray<Qwen3_4B> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn apply_chat_template_and_tokenize(
        prompt: String,
        enable_thinking: bool,
    ) -> TokenArray<Qwen3_4B> {
        let prompt_with_template = build_simple_qwen3_chatml_template(&prompt, enable_thinking);
        Self::tokenize(prompt_with_template)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&QWEN3_4B_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&QWEN3_4B_TOKENIZER, token_ids)
    }

    fn eos_token_id() -> i32 {
        token_to_i32_id(&QWEN3_4B_TOKENIZER, "<|im_end|>", Qwen3_4B::API_NAME)
    }
}

impl LlmModelMarker for Qwen3_4B {
    type Tokenizer = Qwen3_4BTokenizer;
    type Callable = Qwen3_4BLlmCallable;

    const CLI_NAME: &'static str = "qwen3-4b";
    const API_NAME: &'static str = "Qwen/Qwen3-4B";
}
