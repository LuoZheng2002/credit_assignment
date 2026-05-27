use async_trait::async_trait;
use clap::Args;
use reqwest::Client;

pub mod gpt_4o;
pub mod llm_model_name;
pub mod qwen2_5_7b;
pub mod qwen3_4b;
pub mod qwen3_5_0_8b;
pub mod qwen3_5_4b;
pub mod qwen_shared;

use crate::token_array::TokenArray;
pub use crate::token_array::{TokenArrayWithLogprob, TokenLogprobCandidate, Top8Candidates};
pub use gpt_4o::{Gpt4o, Gpt4oLlmCallable, Gpt4oTokenizer};
pub use llm_model_name::LlmModelName;
pub use qwen2_5_7b::{Qwen25, Qwen25LlmCallable, Qwen25Tokenizer};
pub use qwen3_4b::{Qwen3_4B, Qwen3_4BLlmCallable, Qwen3_4BTokenizer};
pub use qwen3_5_0_8b::{Qwen35_08B, Qwen35_08BLlmCallable, Qwen35_08BTokenizer};
pub use qwen3_5_4b::{Qwen35_4B, Qwen35_4BLlmCallable, Qwen35_4BTokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmFamily {
    Gpt,
    Qwen,
}

pub trait MyTokenizer<M: LlmModelMarker>: Send + Sync + 'static {
    fn tokenize(prompt: String) -> TokenArray<M>;
    fn tokenize_prompt_for_generation(prompt: String) -> TokenArray<M> {
        Self::tokenize(prompt)
    }
    fn encode_to_i32_ids(text: &str) -> Vec<i32>;
    fn decode_i32_ids(token_ids: &[i32]) -> String;
    fn eos_token_id() -> i32;
}

#[async_trait]
pub trait LlmCallable<M: LlmModelMarker>: Clone + Send + Sync {
    async fn generate_tokens(
        &self,
        tokens: Vec<i32>,
        passes_in_stop: bool,
    ) -> Result<Vec<i32>, String>;

    async fn generate_tokens_with_logprobs(
        &self,
        _tokens: Vec<i32>,
        _passes_in_stop: bool,
        _temperature: f32,
        _trim_eos: bool,
    ) -> Result<TokenArrayWithLogprob<M>, String> {
        panic!(
            "generate_tokens_with_logprobs is only implemented for callables that support logprobs"
        )
    }
}

pub(crate) fn trim_tail_eos_if_needed<M: LlmModelMarker>(
    mut output: TokenArrayWithLogprob<M>,
    trim_eos: bool,
) -> TokenArrayWithLogprob<M> {
    if !trim_eos {
        return output;
    }

    let eos_token_id = <M::Tokenizer as MyTokenizer<M>>::eos_token_id();
    let Some(&last_token_id) = output.tokens.last() else {
        return output;
    };

    if last_token_id == eos_token_id {
        if output.tokens.len() == 1 {
            return output;
        }

        assert!(
            output.tokens[..output.tokens.len() - 1]
                .iter()
                .all(|&token_id| token_id != eos_token_id),
            "trim_eos=true requires non-tail generated tokens to all be non-EOS",
        );

        output.tokens.pop();
        output.logprobs.pop();
        return output;
    }

    assert!(
        output
            .tokens
            .iter()
            .all(|&token_id| token_id != eos_token_id),
        "trim_eos=true requires EOS to appear only as an optional tail token",
    );
    output
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

    fn tokenize(prompt: String) -> TokenArray<Self> {
        <Self::Tokenizer as MyTokenizer<Self>>::tokenize(prompt)
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable;
}

#[derive(Args, Clone, Debug)]
pub struct LlmCliArgs {
    #[arg(long)]
    pub model_cli_name: String,
    #[arg(long)]
    pub qwen_sglang_port: Option<u16>,
    #[arg(long, default_value_t = 100)]
    pub max_concurrent_requests: usize,
}

impl LlmCliArgs {
    pub fn qwen_sglang_port(&self) -> u16 {
        let port = self
            .qwen_sglang_port
            .expect("Qwen SGLang backend requires --qwen-sglang-port");
        assert!(port > 0, "SGLang port must be greater than 0");
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
        LlmModelName::Qwen35_08b => Qwen35_08B::build_prefix_thinking_disabled(
            prompt_before_assistant,
            prompt_after_assistant,
        ),
        LlmModelName::Gpt4o => {
            Gpt4o::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
        }
    }
}
