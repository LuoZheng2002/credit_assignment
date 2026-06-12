use minijinja::context;
use serde::Serialize;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::{token_array::TokenArray, utils::load_jinja_template_environment};

use super::sglang_model_shared::{
    build_qwen25_python_response_turn_disable_thinking,
    build_qwen25_python_response_turn_enable_thinking, decode_from_i32_ids, encode_to_i32_ids,
    token_to_i32_id,
};
use super::{LlmModelMarker, MyTokenizer, SglangLlmCallable};

static QWEN25_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen25_7B::API_NAME, None).unwrap());

static QWEN25_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment<'static>> =
    LazyLock::new(|| {
        load_jinja_template_environment("tokenizers/qwen25/chat_template.jinja", "chat").unwrap()
    });

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

fn build_qwen25_chat_template(user_prompt: &str, enable_thinking: bool) -> String {
    let tmpl = QWEN25_TEMPLATE_ENVIRONMENT.get_template("chat").unwrap();
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: user_prompt.into(),
    }];
    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
    })
    .unwrap()
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Qwen25_7B;

pub struct Qwen25Tokenizer;
impl MyTokenizer<Qwen25_7B> for Qwen25Tokenizer {
    fn tokenize(prompt: String) -> TokenArray<Qwen25_7B> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn apply_chat_template_and_tokenize(
        prompt: String,
        enable_thinking: bool,
    ) -> TokenArray<Qwen25_7B> {
        let prompt_with_template = build_qwen25_chat_template(&prompt, enable_thinking);
        Self::tokenize(prompt_with_template)
    }

    fn apply_python_response_template_and_tokenize(
        raw_python_response: String,
        enable_thinking: bool,
    ) -> TokenArray<Qwen25_7B> {
        let wrapped_turn = if enable_thinking {
            build_qwen25_python_response_turn_enable_thinking(&raw_python_response)
        } else {
            build_qwen25_python_response_turn_disable_thinking(&raw_python_response)
        };
        Self::tokenize(wrapped_turn)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&QWEN25_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&QWEN25_TOKENIZER, token_ids)
    }

    fn eos_token_id() -> i32 {
        token_to_i32_id(&QWEN25_TOKENIZER, "<|im_end|>", Qwen25_7B::API_NAME)
    }
}

impl LlmModelMarker for Qwen25_7B {
    type Tokenizer = Qwen25Tokenizer;
    type Callable = SglangLlmCallable<Self>;

    const CLI_NAME: &'static str = "qwen2.5-7b";
    const API_NAME: &'static str = "Qwen/Qwen2.5-7B-Instruct";
    const MODEL_LABEL: &'static str = "Qwen2.5-7B model";
}
