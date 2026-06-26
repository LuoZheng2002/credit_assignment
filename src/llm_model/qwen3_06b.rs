use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::utils::load_tokenizer_from_local_or_hf;

use super::sglang_model_shared::{
    build_qwen3_python_response_turn_disable_thinking,
    build_qwen3_python_response_turn_enable_thinking, decode_from_i32_ids, encode_to_i32_ids,
    token_to_i32_id,
};
use super::{
    LlmModelMarker, MyTokenizer, SglangLlmCallable, TokenArray, build_simple_qwen3_chatml_template,
};

static QWEN3_06B_TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(|| {
    load_tokenizer_from_local_or_hf("tokenizers/qwen3/tokenizer.json", Qwen3_06B::API_NAME)
});

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Qwen3_06B;

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
    type Callable = SglangLlmCallable<Self>;

    const CLI_NAME: &'static str = "qwen3-0.6b";
    const API_NAME: &'static str = "Qwen/Qwen3-0.6B";
    const MODEL_LABEL: &'static str = "Qwen3-0.6B model";
}
