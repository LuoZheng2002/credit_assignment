use clap::Args;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::apply_vllm_model_chat_template::apply_vllm_model_chat_template;
use crate::call_llm::{
    GptLlmCallable, LlmCallable, Qwen25LlmCallable, Qwen35LlmCallable, Qwen3LlmCallable,
};
use crate::llm_model_name::LlmModelName;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmFamily {
    Gpt,
    Qwen,
}

pub trait LlmModelMarker: Sized + Send + Sync + 'static {
    type StringOrTokenArray:
        Serialize + for<'de> Deserialize<'de> + Clone + std::fmt::Debug + Send + Sync + 'static;
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

    fn tokenize(prompt: String) -> Self::StringOrTokenArray;

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

static QWEN25_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen25::API_NAME, None).unwrap());
static QWEN3_4B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen3_4B::API_NAME, None).unwrap());
static QWEN3_8B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen3_8B::API_NAME, None).unwrap());
static QWEN35_4B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen35_4B::API_NAME, None).unwrap());

fn tokenize_prompt_for_qwen_model<M: LlmModelMarker>(prompt: &str) -> Vec<i32> {
    let tokenizer = match M::CLI_NAME {
        Qwen25::CLI_NAME => &*QWEN25_TOKENIZER,
        Qwen3_4B::CLI_NAME => &*QWEN3_4B_TOKENIZER,
        Qwen3_8B::CLI_NAME => &*QWEN3_8B_TOKENIZER,
        Qwen35_4B::CLI_NAME => &*QWEN35_4B_TOKENIZER,
        _ => panic!(
            "tokenize_prompt_for_qwen_model called for non-Qwen marker {}",
            M::CLI_NAME
        ),
    };
    tokenizer
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|token| i32::try_from(*token).expect("token id must fit in i32"))
        .collect()
}
pub struct Qwen25;
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Qwen25TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Qwen3TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Qwen35TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}
impl LlmModelMarker for Qwen25 {
    type StringOrTokenArray = Qwen25TokenArray;
    type Callable = Qwen25LlmCallable;

    const CLI_NAME: &'static str = "qwen2.5-7b";
    const API_NAME: &'static str = "Qwen/Qwen2.5-7B-Instruct";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt =
            apply_vllm_model_chat_template(LlmModelName::Qwen25_7b, prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        apply_vllm_model_chat_template(LlmModelName::Qwen25_7b, prompt_before_assistant, true)
    }

    fn tokenize(prompt: String) -> Self::StringOrTokenArray {
        let tokens = tokenize_prompt_for_qwen_model::<Self>(&prompt);
        Qwen25TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable {
        Qwen25LlmCallable::new(
            client,
            llm_cli_args.single_port_for_qwen(),
            llm_cli_args.max_concurrent_requests,
        )
    }
}
pub struct Qwen3_4B;
impl LlmModelMarker for Qwen3_4B {
    type StringOrTokenArray = Qwen3TokenArray;
    type Callable = Qwen3LlmCallable;

    const CLI_NAME: &'static str = "qwen3-4b";
    const API_NAME: &'static str = "Qwen/Qwen3-4B";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt =
            apply_vllm_model_chat_template(LlmModelName::Qwen3_4b, prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        apply_vllm_model_chat_template(LlmModelName::Qwen3_4b, prompt_before_assistant, true)
    }

    fn tokenize(prompt: String) -> Self::StringOrTokenArray {
        let tokens = tokenize_prompt_for_qwen_model::<Self>(&prompt);
        Qwen3TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable {
        Qwen3LlmCallable::new(
            client,
            Self::API_NAME,
            llm_cli_args.single_port_for_qwen(),
            llm_cli_args.max_concurrent_requests,
        )
    }
}
pub struct Qwen3_8B;
impl LlmModelMarker for Qwen3_8B {
    type StringOrTokenArray = Qwen3TokenArray;
    type Callable = Qwen3LlmCallable;

    const CLI_NAME: &'static str = "qwen3-8b";
    const API_NAME: &'static str = "Qwen/Qwen3-8B";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt =
            apply_vllm_model_chat_template(LlmModelName::Qwen3_8b, prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        apply_vllm_model_chat_template(LlmModelName::Qwen3_8b, prompt_before_assistant, true)
    }

    fn tokenize(prompt: String) -> Self::StringOrTokenArray {
        let tokens = tokenize_prompt_for_qwen_model::<Self>(&prompt);
        Qwen3TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable {
        Qwen3LlmCallable::new(
            client,
            Self::API_NAME,
            llm_cli_args.single_port_for_qwen(),
            llm_cli_args.max_concurrent_requests,
        )
    }
}
pub struct Qwen35_4B;
impl LlmModelMarker for Qwen35_4B {
    type StringOrTokenArray = Qwen35TokenArray;
    type Callable = Qwen35LlmCallable;

    const CLI_NAME: &'static str = "qwen3.5-4b";
    const API_NAME: &'static str = "Qwen/Qwen3.5-4B";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt =
            apply_vllm_model_chat_template(LlmModelName::Qwen35_4b, prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        apply_vllm_model_chat_template(LlmModelName::Qwen35_4b, prompt_before_assistant, true)
    }

    fn tokenize(prompt: String) -> Self::StringOrTokenArray {
        let tokens = tokenize_prompt_for_qwen_model::<Self>(&prompt);
        Qwen35TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }

    fn callable_from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self::Callable {
        Qwen35LlmCallable::new(
            client,
            llm_cli_args.single_port_for_qwen(),
            llm_cli_args.max_concurrent_requests,
        )
    }
}
pub struct Gpt4o;
impl LlmModelMarker for Gpt4o {
    type StringOrTokenArray = String;
    type Callable = GptLlmCallable<Self>;

    const CLI_NAME: &'static str = "gpt-4o";
    const API_NAME: &'static str = "gpt-4o";
    const FAMILY: LlmFamily = LlmFamily::Gpt;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        format!("{}\nAssistant: {}", prompt_before_assistant, prompt_after_assistant)
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        prompt_before_assistant.to_string()
    }

    fn tokenize(prompt: String) -> Self::StringOrTokenArray {
        prompt
    }

    fn callable_from_cli_args(client: Client, _llm_cli_args: &LlmCliArgs) -> Self::Callable {
        GptLlmCallable::new(client)
    }
}

pub struct Gpt5Mini;
impl LlmModelMarker for Gpt5Mini {
    type StringOrTokenArray = String;
    type Callable = GptLlmCallable<Self>;

    const CLI_NAME: &'static str = "gpt-5-mini";
    const API_NAME: &'static str = "gpt-5-mini";
    const FAMILY: LlmFamily = LlmFamily::Gpt;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        format!("{}\nAssistant: {}", prompt_before_assistant, prompt_after_assistant)
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        prompt_before_assistant.to_string()
    }

    fn tokenize(prompt: String) -> Self::StringOrTokenArray {
        prompt
    }

    fn callable_from_cli_args(client: Client, _llm_cli_args: &LlmCliArgs) -> Self::Callable {
        GptLlmCallable::new(client)
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
        LlmModelName::Qwen3_8b => {
            Qwen3_8B::build_prefix_thinking_disabled(prompt_before_assistant, prompt_after_assistant)
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
