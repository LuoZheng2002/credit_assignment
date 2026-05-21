use async_trait::async_trait;
use minijinja::context;
use reqwest::Client;
use serde::Serialize;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::token_array::TokenArray;

use super::qwen_shared::{
    SharedQwenLlmCallable, decode_from_i32_ids, encode_to_i32_ids, token_to_i32_id,
};
use super::{LlmCallable, LlmCliArgs, LlmFamily, LlmModelMarker, MyTokenizer};

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
    pub fn new(client: Client, vllm_port: u16, max_concurrent_requests: usize) -> Self {
        Self {
            shared: SharedQwenLlmCallable::new(
                client,
                Qwen25::API_NAME,
                vllm_port,
                max_concurrent_requests,
            ),
        }
    }
}

#[async_trait]
impl LlmCallable<Qwen25> for Qwen25LlmCallable {
    async fn generate_text(&self, prompt_or_tokens: Vec<i32>, passes_in_stop: bool) -> String {
        self.shared
            .generate_from_tokens(prompt_or_tokens, passes_in_stop)
            .await
    }
}

pub struct Qwen25Tokenizer;
impl MyTokenizer<Qwen25> for Qwen25Tokenizer {
    fn tokenize(prompt: String) -> TokenArray {
        let tokens = Self::encode_to_i32_ids(&prompt);
        TokenArray {
            tokens,
            decoded_string: prompt,
        }
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
        Qwen25LlmCallable::new(
            client,
            llm_cli_args.single_port_for_qwen(),
            llm_cli_args.max_concurrent_requests,
        )
    }
}
