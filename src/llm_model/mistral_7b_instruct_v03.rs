use async_trait::async_trait;
use minijinja::context;
use reqwest::Client;
use serde::Serialize;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::utils::load_jinja_template_environment;

use super::sglang_model_shared::{
    SharedSglangLlmCallable, decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id,
    wrap_python_response_xml,
};
use super::{
    LlmCallable, LlmCliArgs, LlmModelMarker, MyTokenizer, TokenArray, TokenArrayWithLogprob,
    trim_tail_eos_if_needed,
};

static MISTRAL_7B_INSTRUCT_V03_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Mistral7BInstructV03::API_NAME, None).unwrap());

static MISTRAL_7B_INSTRUCT_V03_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment<'static>> =
    LazyLock::new(|| {
        load_jinja_template_environment("tokenizers/mistral7b/chat_template.jinja", "chat")
            .unwrap()
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

#[derive(Clone)]
pub struct Mistral7BInstructV03LlmCallable {
    shared: SharedSglangLlmCallable,
}

#[async_trait]
impl LlmCallable<Mistral7BInstructV03> for Mistral7BInstructV03LlmCallable {
    fn from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self {
        Self {
            shared: SharedSglangLlmCallable::from_llm_cli_args(
                client,
                llm_cli_args,
                "Mistral-7B-Instruct-v0.3 model",
            ),
        }
    }

    async fn generate_tokens(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> Result<Vec<i32>, String> {
        self.shared
            .generate_tokens_from_tokens::<Mistral7BInstructV03>(prompt_or_tokens, passes_in_stop)
            .await
    }

    async fn generate_tokens_with_logprobs(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
        trim_eos: bool,
    ) -> Result<TokenArrayWithLogprob<Mistral7BInstructV03>, String> {
        let output = self
            .shared
            .generate_tokens_with_logprobs_from_tokens(
                prompt_or_tokens,
                passes_in_stop,
                temperature,
            )
            .await?;
        Ok(trim_tail_eos_if_needed::<Mistral7BInstructV03>(
            output, trim_eos,
        ))
    }
}

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
    type Callable = Mistral7BInstructV03LlmCallable;

    const CLI_NAME: &'static str = "mistral-7b-instruct-v0.3";
    const API_NAME: &'static str = "mistralai/Mistral-7B-Instruct-v0.3";
}
