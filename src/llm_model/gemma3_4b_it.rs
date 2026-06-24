use minijinja::context;
use serde::Serialize;
use std::{env, sync::LazyLock};
use tokenizers::Tokenizer;

use crate::utils::load_jinja_template_environment;

use super::sglang_model_shared::{
    decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id, wrap_python_response_xml,
};
use super::{LlmModelMarker, MyTokenizer, SglangLlmCallable, TokenArray};

static GEMMA3_4B_IT_TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(|| {
    let token = env::var("HF_TOKEN")
        .ok()
        .or_else(|| env::var("HUGGINGFACE_HUB_TOKEN").ok());
    let params = tokenizers::FromPretrainedParameters {
        token,
        ..Default::default()
    };
    Tokenizer::from_pretrained(Gemma3_4BIt::API_NAME, Some(params)).unwrap()
});

static GEMMA3_4B_IT_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment<'static>> =
    LazyLock::new(|| {
        load_jinja_template_environment("tokenizers/gemma3/chat_template.jinja", "chat").unwrap()
    });

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

fn build_gemma3_4b_it_chat_template(user_prompt: &str) -> String {
    let tmpl = GEMMA3_4B_IT_TEMPLATE_ENVIRONMENT
        .get_template("chat")
        .unwrap();
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: user_prompt.into(),
    }];
    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => true,
        bos_token => "<bos>",
    })
    .unwrap()
}

fn build_gemma3_4b_it_python_response_turn(raw_python_response: &str) -> String {
    let wrapped_response = wrap_python_response_xml(raw_python_response);
    format!(
        "<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        wrapped_response
    )
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Gemma3_4BIt;

pub struct Gemma3_4BItTokenizer;
impl MyTokenizer<Gemma3_4BIt> for Gemma3_4BItTokenizer {
    fn tokenize(prompt: String) -> TokenArray<Gemma3_4BIt> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn apply_chat_template_and_tokenize(
        prompt: String,
        enable_thinking: bool,
    ) -> TokenArray<Gemma3_4BIt> {
        let _ = enable_thinking;
        let prompt_with_template = build_gemma3_4b_it_chat_template(&prompt);
        Self::tokenize(prompt_with_template)
    }

    fn apply_python_response_template_and_tokenize(
        raw_python_response: String,
        enable_thinking: bool,
    ) -> TokenArray<Gemma3_4BIt> {
        let _ = enable_thinking;
        let wrapped_turn = build_gemma3_4b_it_python_response_turn(&raw_python_response);
        Self::tokenize(wrapped_turn)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&GEMMA3_4B_IT_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&GEMMA3_4B_IT_TOKENIZER, token_ids)
    }

    fn eos_token_id() -> i32 {
        token_to_i32_id(
            &GEMMA3_4B_IT_TOKENIZER,
            "<end_of_turn>",
            Gemma3_4BIt::API_NAME,
        )
    }
}

impl LlmModelMarker for Gemma3_4BIt {
    type Tokenizer = Gemma3_4BItTokenizer;
    type Callable = SglangLlmCallable<Self>;

    const CLI_NAME: &'static str = "gemma-3-4b-it";
    const API_NAME: &'static str = "google/gemma-3-4b-it";
    const MODEL_LABEL: &'static str = "Gemma-3-4B-IT model";
}
