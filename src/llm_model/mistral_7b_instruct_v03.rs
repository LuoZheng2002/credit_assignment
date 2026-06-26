use minijinja::context;
use serde::Serialize;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::utils::{load_jinja_template_environment, load_tokenizer_from_local_or_hf};

use super::sglang_model_shared::{
    decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id, wrap_python_response_xml,
};
use super::{LlmModelMarker, MyTokenizer, SglangLlmCallable, TokenArray};

static MISTRAL_7B_INSTRUCT_V03_TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(|| {
    load_tokenizer_from_local_or_hf(
        "tokenizers/mistral7b/tokenizer.json",
        Mistral7BInstructV03::API_NAME,
    )
});

static MISTRAL_7B_INSTRUCT_V03_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment<'static>> =
    LazyLock::new(|| {
        load_jinja_template_environment("tokenizers/mistral7b/chat_template.jinja", "chat").unwrap()
    });

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

fn build_mistral_7b_instruct_v03_chat_template(user_prompt: &str) -> String {
    let tmpl = MISTRAL_7B_INSTRUCT_V03_TEMPLATE_ENVIRONMENT
        .get_template("chat")
        .unwrap();
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: user_prompt.into(),
    }];
    tmpl.render(context! {
        messages => messages,
        bos_token => "<s>",
        eos_token => "</s>",
    })
    .unwrap()
}

fn build_mistral_7b_instruct_v03_python_response_turn(raw_python_response: &str) -> String {
    let wrapped_response = wrap_python_response_xml(raw_python_response);
    format!("</s>[INST] {}[/INST]", wrapped_response)
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Mistral7BInstructV03;

pub struct Mistral7BInstructV03Tokenizer;
impl MyTokenizer<Mistral7BInstructV03> for Mistral7BInstructV03Tokenizer {
    fn tokenize(prompt: String) -> TokenArray<Mistral7BInstructV03> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn apply_chat_template_and_tokenize(
        prompt: String,
        enable_thinking: bool,
    ) -> TokenArray<Mistral7BInstructV03> {
        let _ = enable_thinking;
        let prompt_with_template = build_mistral_7b_instruct_v03_chat_template(&prompt);
        Self::tokenize(prompt_with_template)
    }

    fn apply_python_response_template_and_tokenize(
        raw_python_response: String,
        enable_thinking: bool,
    ) -> TokenArray<Mistral7BInstructV03> {
        let _ = enable_thinking;
        let wrapped_turn = build_mistral_7b_instruct_v03_python_response_turn(&raw_python_response);
        Self::tokenize(wrapped_turn)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&MISTRAL_7B_INSTRUCT_V03_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&MISTRAL_7B_INSTRUCT_V03_TOKENIZER, token_ids)
    }

    fn eos_token_id() -> i32 {
        token_to_i32_id(
            &MISTRAL_7B_INSTRUCT_V03_TOKENIZER,
            "</s>",
            Mistral7BInstructV03::API_NAME,
        )
    }
}

impl LlmModelMarker for Mistral7BInstructV03 {
    type Tokenizer = Mistral7BInstructV03Tokenizer;
    type Callable = SglangLlmCallable<Self>;

    const CLI_NAME: &'static str = "mistral";
    const API_NAME: &'static str = "mistralai/Mistral-7B-Instruct-v0.3";
    const MODEL_LABEL: &'static str = "Mistral-7B-Instruct-v0.3 model";
}
