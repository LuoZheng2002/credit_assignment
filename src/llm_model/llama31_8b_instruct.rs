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

static LLAMA31_8B_INSTRUCT_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Llama31_8BInstruct::API_NAME, None).unwrap());

static LLAMA31_8B_INSTRUCT_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment> =
    LazyLock::new(|| {
        let mut env = minijinja::Environment::new();
        let template_src =
            std::fs::read_to_string("tokenizers/llama31/chat_template.jinja").unwrap();
        env.add_template_owned("chat", template_src).unwrap();
        env
    });

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

fn build_llama31_8b_instruct_chat_template(user_prompt: &str) -> String {
    let tmpl = LLAMA31_8B_INSTRUCT_TEMPLATE_ENVIRONMENT
        .get_template("chat")
        .unwrap();
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: user_prompt.into(),
    }];
    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => true,
        bos_token => "<|begin_of_text|>",
    })
    .unwrap()
}

fn build_llama31_8b_instruct_python_response_turn(raw_python_response: &str) -> String {
    let wrapped_response = wrap_python_response_xml(raw_python_response);
    format!(
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        wrapped_response
    )
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Llama31_8BInstruct;

#[derive(Clone)]
pub struct Llama31_8BInstructLlmCallable {
    shared: SharedSglangLlmCallable,
}

#[async_trait]
impl LlmCallable<Llama31_8BInstruct> for Llama31_8BInstructLlmCallable {
    fn from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self {
        Self {
            shared: SharedSglangLlmCallable::from_llm_cli_args(
                client,
                llm_cli_args,
                "Llama-3.1-8B-Instruct model",
            ),
        }
    }

    async fn generate_tokens(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> Result<Vec<i32>, String> {
        self.shared
            .generate_tokens_from_tokens::<Llama31_8BInstruct>(prompt_or_tokens, passes_in_stop)
            .await
    }

    async fn generate_tokens_with_logprobs(
        &self,
        prompt_or_tokens: Vec<i32>,
        passes_in_stop: bool,
        temperature: f32,
        trim_eos: bool,
    ) -> Result<TokenArrayWithLogprob<Llama31_8BInstruct>, String> {
        let output = self
            .shared
            .generate_tokens_with_logprobs_from_tokens(
                prompt_or_tokens,
                passes_in_stop,
                temperature,
            )
            .await?;
        Ok(trim_tail_eos_if_needed::<Llama31_8BInstruct>(
            output, trim_eos,
        ))
    }
}

pub struct Llama31_8BInstructTokenizer;
impl MyTokenizer<Llama31_8BInstruct> for Llama31_8BInstructTokenizer {
    fn tokenize(prompt: String) -> TokenArray<Llama31_8BInstruct> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn apply_chat_template_and_tokenize(
        prompt: String,
        enable_thinking: bool,
    ) -> TokenArray<Llama31_8BInstruct> {
        let _ = enable_thinking;
        let prompt_with_template = build_llama31_8b_instruct_chat_template(&prompt);
        Self::tokenize(prompt_with_template)
    }

    fn apply_python_response_template_and_tokenize(
        raw_python_response: String,
        enable_thinking: bool,
    ) -> TokenArray<Llama31_8BInstruct> {
        let _ = enable_thinking;
        let wrapped_turn = build_llama31_8b_instruct_python_response_turn(&raw_python_response);
        Self::tokenize(wrapped_turn)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&LLAMA31_8B_INSTRUCT_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&LLAMA31_8B_INSTRUCT_TOKENIZER, token_ids)
    }

    fn eos_token_id() -> i32 {
        token_to_i32_id(
            &LLAMA31_8B_INSTRUCT_TOKENIZER,
            "<|eot_id|>",
            Llama31_8BInstruct::API_NAME,
        )
    }
}

impl LlmModelMarker for Llama31_8BInstruct {
    type Tokenizer = Llama31_8BInstructTokenizer;
    type Callable = Llama31_8BInstructLlmCallable;

    const CLI_NAME: &'static str = "llama-3.1-8b-instruct";
    const API_NAME: &'static str = "meta-llama/Llama-3.1-8B-Instruct";
}
