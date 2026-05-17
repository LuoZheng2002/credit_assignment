use async_trait::async_trait;
use clap::Args;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub mod gpt_4o;
pub mod gpt_5_mini;
pub mod llm_model_name;
pub mod qwen2_5_7b;
pub mod qwen_shared;
pub mod qwen3_4b;
pub mod qwen3_5_4b;

pub use llm_model_name::LlmModelName;
pub use gpt_4o::{Gpt4o, Gpt4oLlmCallable};
pub use gpt_5_mini::{Gpt5Mini, Gpt5MiniLlmCallable};
pub use qwen2_5_7b::{Qwen25, Qwen25LlmCallable, Qwen25TokenArray, Qwen25Tokenizer};
pub use qwen3_4b::{Qwen3_4B, Qwen3_4BLlmCallable, Qwen3TokenArray};
pub use qwen3_5_4b::{Qwen35_4B, Qwen35_4BLlmCallable, Qwen35TokenArray};
pub use qwen_shared::CONTEXT_LENGTH_EXCEEDED_RESPONSE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmFamily {
    Gpt,
    Qwen,
}

pub trait MyTokenizer<M: LlmModelMarker>: Send + Sync + 'static {
    fn tokenize(prompt: String) -> M::StringOrTokenArray;
    fn encode_to_i32_ids(text: &str) -> Vec<i32>;
    fn token_to_id(token: &str) -> i32;
}

#[async_trait]
pub trait LlmCallable<M: LlmModelMarker>: Clone + Send + Sync {
    async fn generate(
        &self,
        prompt_or_tokens: M::StringOrTokenArray,
        passes_in_stop: bool,
    ) -> String;

    async fn call_with_prefix_thinking_disabled(
        &self,
        prompt_before_assistant: String,
        prompt_after_assistant: String,
    ) -> String {
        let prompt =
            M::build_prefix_thinking_disabled(&prompt_before_assistant, &prompt_after_assistant);
        let input = M::tokenize(prompt);
        self.generate(input, true).await
    }

    async fn call_with_prefix_thinking_enabled(&self, prompt_before_assistant: String) -> String {
        let prompt = M::build_prefix_thinking_enabled(&prompt_before_assistant);
        let input = M::tokenize(prompt);
        self.generate(input, true).await
    }
}

pub trait LlmModelMarker: Sized + Send + Sync + 'static {
    type StringOrTokenArray:
        Serialize + for<'de> Deserialize<'de> + Clone + std::fmt::Debug + Send + Sync + 'static;
    type Tokenizer: MyTokenizer<Self>;
    type Callable: LlmCallable<Self> + Send + Sync + 'static;

    const CLI_NAME: &'static str;
    const API_NAME: &'static str;
    const FAMILY: LlmFamily;

    fn is_qwen() -> bool {
        matches!(Self::FAMILY, LlmFamily::Qwen)
    }

    fn is_gpt() -> bool {
        matches!(Self::FAMILY, LlmFamily::Gpt)
    }

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String;

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String;

    fn tokenize(prompt: String) -> Self::StringOrTokenArray {
        <Self::Tokenizer as MyTokenizer<Self>>::tokenize(prompt)
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable;
}

#[derive(Args, Clone, Debug)]
pub struct LlmCliArgs {
    #[arg(long)]
    pub model_cli_name: String,
    #[arg(long)]
    pub gpt_vllm_port: Option<u16>,
    #[arg(long)]
    pub qwen_vllm_port: Option<u16>,
    #[arg(long, default_value_t = 100)]
    pub max_concurrent_requests: usize,
}

impl LlmCliArgs {
    pub fn single_port_for_qwen(&self) -> u16 {
        let port = self
            .qwen_vllm_port
            .or(self.gpt_vllm_port)
            .expect("Qwen model requires --qwen-vllm-port (or --gpt-vllm-port as fallback)");
        assert!(port > 0, "vLLM port must be greater than 0");
        port
    }
}

pub(crate) fn build_simple_qwen_chatml_prefix(user_prompt: &str, enable_thinking: bool) -> String {
    if enable_thinking {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            user_prompt
        )
    } else {
        format!(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_prompt
        )
    }
}

pub struct PassthroughTokenizer;
impl<M> MyTokenizer<M> for PassthroughTokenizer
where
    M: LlmModelMarker<StringOrTokenArray = String>,
{
    fn tokenize(prompt: String) -> M::StringOrTokenArray {
        prompt
    }

    fn encode_to_i32_ids(text: &str) -> Vec<i32> {
        println!(
            "Warning: encode_to_i32_ids called for GPT model {}; falling back to {} tokenizer",
            M::CLI_NAME,
            Qwen25::CLI_NAME
        );
        crate::llm_model::qwen2_5_7b::Qwen25Tokenizer::encode_to_i32_ids(text)
    }

    fn token_to_id(token: &str) -> i32 {
        println!(
            "Warning: token_to_id('{token}') called for GPT model {}; returning 0",
            M::CLI_NAME
        );
        0
    }
}

pub type Qwen3 = Qwen3_4B;
pub type Qwen35 = Qwen35_4B;

pub fn build_prefix_thinking_disabled_by_model(
    model: LlmModelName,
    prompt_before_assistant: &str,
    prompt_after_assistant: &str,
) -> String {
    match model {
        LlmModelName::Qwen25_7b => {
            Qwen25::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
        }
        LlmModelName::Qwen3_4b => {
            Qwen3_4B::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
        }
        LlmModelName::Qwen35_4b => {
            Qwen35_4B::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
        }
        LlmModelName::Gpt4o => {
            Gpt4o::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
        }
        LlmModelName::Gpt5Mini => {
            Gpt5Mini::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
        }
    }
}
