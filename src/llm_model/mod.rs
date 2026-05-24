use async_trait::async_trait;
use clap::Args;
use reqwest::Client;

pub mod gpt_4o;
pub mod gpt_5_mini;
pub mod llm_model_name;
pub mod qwen2_5_7b;
pub mod qwen3_4b;
pub mod qwen3_5_4b;
pub mod qwen_shared;

use crate::token_array::TokenArray;
pub use crate::token_array::{TokenArrayWithLogprob, TokenLogprobCandidate, Top8Candidates};
pub use gpt_4o::{Gpt4o, Gpt4oLlmCallable, Gpt4oTokenizer};
pub use gpt_5_mini::{Gpt5Mini, Gpt5MiniLlmCallable, Gpt5MiniTokenizer};
pub use llm_model_name::LlmModelName;
pub use qwen_shared::CONTEXT_LENGTH_EXCEEDED_RESPONSE;
pub use qwen2_5_7b::{Qwen25, Qwen25LlmCallable, Qwen25Tokenizer};
pub use qwen3_4b::{Qwen3_4B, Qwen3_4BLlmCallable, Qwen3_4BTokenizer};
pub use qwen3_5_4b::{Qwen35_4B, Qwen35_4BLlmCallable, Qwen35_4BTokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmFamily {
    Gpt,
    Qwen,
}

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
pub enum QwenApiBackend {
    Vllm,
    Sglang,
    VllmWrapper,
    Openrouter,
}

pub trait MyTokenizer<M: LlmModelMarker>: Send + Sync + 'static {
    fn tokenize(prompt: String) -> TokenArray;
    fn encode_to_i32_ids(text: &str) -> Vec<i32>;
    fn decode_i32_ids(token_ids: &[i32]) -> String;
    fn token_to_id(token: &str) -> i32;
}

#[async_trait]
pub trait LlmCallable<M: LlmModelMarker>: Clone + Send + Sync {
    async fn generate_text(&self, tokens: Vec<i32>, passes_in_stop: bool) -> String;

    async fn generate_tokens_with_logprobs(
        &self,
        _tokens: Vec<i32>,
        _passes_in_stop: bool,
    ) -> TokenArrayWithLogprob {
        panic!("generate_tokens_with_logprobs is only implemented for vLLM-backed callables")
    }

    async fn call_with_prefix_thinking_disabled(
        &self,
        prompt_before_assistant: String,
        prompt_after_assistant: String,
    ) -> String {
        let prompt =
            M::build_prefix_thinking_disabled(&prompt_before_assistant, &prompt_after_assistant);
        let input = M::tokenize(prompt).tokens;
        self.generate_text(input, true).await
    }

    async fn call_with_prefix_thinking_enabled(&self, prompt_before_assistant: String) -> String {
        let prompt = M::build_prefix_thinking_enabled(&prompt_before_assistant);
        let input = M::tokenize(prompt).tokens;
        self.generate_text(input, true).await
    }
}

pub trait LlmModelMarker: Sized + Send + Sync + 'static {
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

    fn tokenize(prompt: String) -> TokenArray {
        <Self::Tokenizer as MyTokenizer<Self>>::tokenize(prompt)
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable;
}

#[derive(Args, Clone, Debug)]
pub struct LlmCliArgs {
    #[arg(long)]
    pub model_cli_name: String,
    #[arg(long, default_value = "vllm")]
    pub qwen_api_backend: QwenApiBackend,
    #[arg(long)]
    pub qwen_vllm_port: Option<u16>,
    #[arg(long)]
    pub qwen_sglang_port: Option<u16>,
    #[arg(long)]
    pub vllm_wrapper_port: Option<u16>,
    #[arg(long)]
    pub openrouter_model: Option<String>,
    #[arg(long, default_value = "https://openrouter.ai/api/v1")]
    pub openrouter_base_url: String,
    #[arg(long)]
    pub openrouter_http_referer: Option<String>,
    #[arg(long)]
    pub openrouter_x_title: Option<String>,
    #[arg(long, default_value_t = 100)]
    pub max_concurrent_requests: usize,
}

impl LlmCliArgs {
    pub fn qwen_vllm_port(&self) -> u16 {
        let port = self
            .qwen_vllm_port
            .expect("Qwen vLLM backend requires --qwen-vllm-port");
        assert!(port > 0, "vLLM port must be greater than 0");
        port
    }

    pub fn vllm_wrapper_port(&self) -> u16 {
        let port = self
            .vllm_wrapper_port
            .expect("Qwen vLLM wrapper backend requires --vllm-wrapper-port");
        assert!(port > 0, "vLLM wrapper port must be greater than 0");
        port
    }

    pub fn qwen_sglang_port(&self) -> u16 {
        let port = self
            .qwen_sglang_port
            .expect("Qwen SGLang backend requires --qwen-sglang-port");
        assert!(port > 0, "SGLang port must be greater than 0");
        port
    }

    pub fn openrouter_model_or_default(&self, default_model: &'static str) -> String {
        self.openrouter_model
            .clone()
            .unwrap_or_else(|| default_model.to_string())
    }

    pub fn openrouter_api_key(&self) -> String {
        std::env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY environment variable not set for OpenRouter backend")
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
        LlmModelName::Qwen3_4b => Qwen3_4B::build_prefix_thinking_disabled(
            prompt_before_assistant,
            prompt_after_assistant,
        ),
        LlmModelName::Qwen35_4b => Qwen35_4B::build_prefix_thinking_disabled(
            prompt_before_assistant,
            prompt_after_assistant,
        ),
        LlmModelName::Gpt4o => {
            Gpt4o::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
        }
        LlmModelName::Gpt5Mini => Gpt5Mini::build_prefix_thinking_disabled(
            prompt_before_assistant,
            prompt_after_assistant,
        ),
    }
}
