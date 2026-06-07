use async_trait::async_trait;
use minijinja::context;
use reqwest::Client;
use serde::Serialize;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use super::sglang_model_shared::{
    SharedSglangLlmCallable, decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id,
    wrap_python_response_xml,
};
use super::{
    LlmCallable, LlmCliArgs, LlmModelMarker, MyTokenizer, TokenArray, TokenArrayWithLogprob,
    trim_tail_eos_if_needed,
};

static GEMMA3_4B_IT_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Gemma3_4BIt::API_NAME, None).unwrap());

static GEMMA3_4B_IT_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment> = LazyLock::new(|| {
    let mut env = minijinja::Environment::new();
    let template_src = std::fs::read_to_string("tokenizers/gemma3/chat_template.jinja").unwrap();
    env.add_template_owned("chat", template_src).unwrap();
    env
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

#[derive(Clone)]
pub struct Gemma3_4BItLlmCallable {
    shared: SharedSglangLlmCallable,
}

impl Gemma3_4BItLlmCallable {
    pub(crate) fn new(client: Client, sglang_port: u16) -> Self {
        Self {
            shared: SharedSglangLlmCallable::new(client, sglang_port),
        }
    }
}

#[async_trait]
impl LlmCallable<Gemma3_4BIt> for Gemma3_4BItLlmCallable {
    fn from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self {
        let sglang_port = llm_cli_args
            .sglang_port
            .expect("Gemma-3-4B-IT model requires sglang port");
        Gemma3_4BItLlmCallable::new(client, sglang_port)
    }

    async fn generate_tokens(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> Result<Vec<i32>, String> {
        self.shared
            .generate_tokens_from_tokens::<Gemma3_4BIt>(prompt_or_tokens, passes_in_stop)
            .await
    }

    async fn generate_tokens_with_logprobs(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
        trim_eos: bool,
    ) -> Result<TokenArrayWithLogprob<Gemma3_4BIt>, String> {
        let output = self
            .shared
            .generate_tokens_with_logprobs_from_tokens(
                prompt_or_tokens,
                passes_in_stop,
                temperature,
            )
            .await?;
        Ok(trim_tail_eos_if_needed::<Gemma3_4BIt>(output, trim_eos))
    }
}

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
    type Callable = Gemma3_4BItLlmCallable;

    const CLI_NAME: &'static str = "gemma-3-4b-it";
    const API_NAME: &'static str = "google/gemma-3-4b-it";
}
