use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::utils::load_tokenizer_from_local_or_hf;

use super::sglang_model_shared::{
    build_qwen35_python_response_turn_disable_thinking,
    build_qwen35_python_response_turn_enable_thinking, decode_from_i32_ids, encode_to_i32_ids,
    token_to_i32_id,
};
use super::{LlmModelMarker, MyTokenizer, SglangLlmCallable, TokenArray};

static QWEN35_4B_TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(|| {
    load_tokenizer_from_local_or_hf("tokenizers/qwen35/tokenizer.json", Qwen35_4B::API_NAME)
});
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Qwen35_4B;

pub(crate) fn build_simple_qwen35_chatml_template(
    user_prompt: &str,
    enable_thinking: bool,
) -> String {
    if enable_thinking {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            user_prompt
        )
    } else {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            user_prompt
        )
    }
}

pub struct Qwen35_4BTokenizer;
impl MyTokenizer<Qwen35_4B> for Qwen35_4BTokenizer {
    fn tokenize(prompt: String) -> TokenArray<Qwen35_4B> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn apply_chat_template_and_tokenize(
        prompt: String,
        enable_thinking: bool,
    ) -> TokenArray<Qwen35_4B> {
        let prompt_with_template = build_simple_qwen35_chatml_template(&prompt, enable_thinking);
        Self::tokenize(prompt_with_template)
    }

    fn apply_python_response_template_and_tokenize(
        raw_python_response: String,
        enable_thinking: bool,
    ) -> TokenArray<Qwen35_4B> {
        let wrapped_turn = if enable_thinking {
            build_qwen35_python_response_turn_enable_thinking(&raw_python_response)
        } else {
            build_qwen35_python_response_turn_disable_thinking(&raw_python_response)
        };
        Self::tokenize(wrapped_turn)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&QWEN35_4B_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&QWEN35_4B_TOKENIZER, token_ids)
    }

    fn eos_token_id() -> i32 {
        token_to_i32_id(&QWEN35_4B_TOKENIZER, "<|im_end|>", Qwen35_4B::API_NAME)
    }
}

impl LlmModelMarker for Qwen35_4B {
    type Tokenizer = Qwen35_4BTokenizer;
    type Callable = SglangLlmCallable<Self>;

    const CLI_NAME: &'static str = "qwen3.5-4b";
    const API_NAME: &'static str = "Qwen/Qwen3.5-4B";
    const MODEL_LABEL: &'static str = "Qwen3.5-4B model";
}
