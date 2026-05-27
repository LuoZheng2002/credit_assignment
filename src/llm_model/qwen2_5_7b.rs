use async_trait::async_trait;
use minijinja::context;
use reqwest::Client;
use serde::Serialize;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::token_array::TokenArray;

use super::qwen_shared::{
    QwenBackend, SharedQwenLlmCallable, decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id,
};
use super::{
    LlmCallable, LlmCliArgs, LlmFamily, LlmModelMarker, MyTokenizer, TokenArrayWithLogprob,
    trim_tail_eos_if_needed,
};

static QWEN25_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen25::API_NAME, None).unwrap());

static QWEN25_TEMPLATE_ENVIRONMENT: LazyLock<minijinja::Environment> = LazyLock::new(|| {
    let mut env = minijinja::Environment::new();
    let template_src = std::fs::read_to_string("tokenizers/qwen25/chat_template.jinja").unwrap();
    env.add_template_owned("chat", template_src).unwrap();
    env
});

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

fn build_qwen25_prefix(user_prompt: &str, enable_thinking: bool) -> String {
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

pub struct Qwen25;

#[derive(Clone)]
pub struct Qwen25LlmCallable {
    shared: SharedQwenLlmCallable,
}

impl Qwen25LlmCallable {
    pub(crate) fn new(
        client: Client,
        backend: QwenBackend,
        max_concurrent_requests: usize,
    ) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(client, backend, max_concurrent_requests),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen25> for Qwen25LlmCallable {
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
    ) -> Result<TokenArrayWithLogprob<Qwen25>, String> {
        let output = self
            .shared
            .generate_tokens_with_logprobs_from_tokens(
                prompt_or_tokens,
                passes_in_stop,
                temperature,
            )
            .await?;
        Ok(trim_tail_eos_if_needed::<Qwen25>(output, trim_eos))
    }
}

pub struct Qwen25Tokenizer;
impl MyTokenizer<Qwen25> for Qwen25Tokenizer {
    fn tokenize(prompt: String) -> TokenArray<Qwen25> {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray::from_tokens(tokens)
    }

    fn tokenize_prompt_for_generation(prompt: String) -> TokenArray<Qwen25> {
        let prompt_with_template = build_qwen25_prefix(&prompt, false);
        Self::tokenize(prompt_with_template)
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        encode_to_i32_ids(&QWEN25_TOKENIZER, text)
    }

    fn decode_i32_ids(token_ids: &[i32]) -> String {
        decode_from_i32_ids(&QWEN25_TOKENIZER, token_ids)
    }

    fn token_to_id(token: &str) -> i32 {
        token_to_i32_id(&QWEN25_TOKENIZER, token, Qwen25::API_NAME)
    }

    fn eos_token_id() -> i32 {
        Self::token_to_id("<|im_end|>")
    }
}

impl LlmModelMarker for Qwen25 {
    type Tokenizer = Qwen25Tokenizer;
    type Callable = Qwen25LlmCallable;

    const CLI_NAME: &'static str = "qwen2.5-7b";
    const API_NAME: &'static str = "Qwen/Qwen2.5-7B-Instruct";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt = build_qwen25_prefix(prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        build_qwen25_prefix(prompt_before_assistant, true)
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable {
        let backend = QwenBackend {
            sglang_port: llm_cli_args.qwen_sglang_port(),
        };

        Qwen25LlmCallable::new(client, backend, llm_cli_args.max_concurrent_requests)
    }
}
