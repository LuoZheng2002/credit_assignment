use clap::Args;
use minijinja::context;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use tokenizers::Tokenizer;

use crate::call_llm::{
    GptLlmCallable, LlmCallable, Qwen25LlmCallable, Qwen35LlmCallable, Qwen3LlmCallable,
};
use crate::llm_model_name::LlmModelName;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmFamily {
    Gpt,
    Qwen,
}

pub trait MyTokenizer<M: LlmModelMarker>: Send + Sync + 'static {
    fn tokenize(prompt: String) -> M::StringOrTokenArray;
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

static QWEN25_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen25::API_NAME, None).unwrap());
static QWEN3_4B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen3_4B::API_NAME, None).unwrap());
static QWEN35_4B_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_pretrained(Qwen35_4B::API_NAME, None).unwrap());
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

fn build_simple_qwen_chatml_prefix(user_prompt: &str, enable_thinking: bool) -> String {
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

fn tokenize_prompt_for_qwen_model<M: LlmModelMarker>(prompt: &str) -> Vec<i32> {
    let tokenizer = match M::CLI_NAME {
        Qwen25::CLI_NAME => &*QWEN25_TOKENIZER,
        Qwen3_4B::CLI_NAME => &*QWEN3_4B_TOKENIZER,
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

pub struct PassthroughTokenizer;
impl<M> MyTokenizer<M> for PassthroughTokenizer
where
    M: LlmModelMarker<StringOrTokenArray = String>,
{
    fn tokenize(prompt: String) -> M::StringOrTokenArray {
        prompt
    }
}

pub struct Qwen25Tokenizer;
impl MyTokenizer<Qwen25> for Qwen25Tokenizer {
    fn tokenize(prompt: String) -> Qwen25TokenArray {
        let tokens = tokenize_prompt_for_qwen_model::<Qwen25>(&prompt);
        Qwen25TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }
}

pub struct Qwen3_4BTokenizer;
impl MyTokenizer<Qwen3_4B> for Qwen3_4BTokenizer {
    fn tokenize(prompt: String) -> Qwen3TokenArray {
        let tokens = tokenize_prompt_for_qwen_model::<Qwen3_4B>(&prompt);
        Qwen3TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }
}

pub struct Qwen35_4BTokenizer;
impl MyTokenizer<Qwen35_4B> for Qwen35_4BTokenizer {
    fn tokenize(prompt: String) -> Qwen35TokenArray {
        let tokens = tokenize_prompt_for_qwen_model::<Qwen35_4B>(&prompt);
        Qwen35TokenArray {
            tokens,
            decoded_string: prompt,
        }
    }
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
pub struct Qwen3_4B;
impl LlmModelMarker for Qwen3_4B {
    type StringOrTokenArray = Qwen3TokenArray;
    type Tokenizer = Qwen3_4BTokenizer;
    type Callable = Qwen3LlmCallable;

    const CLI_NAME: &'static str = "qwen3-4b";
    const API_NAME: &'static str = "Qwen/Qwen3-4B";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt = build_simple_qwen_chatml_prefix(prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        build_simple_qwen_chatml_prefix(prompt_before_assistant, true)
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
    type Tokenizer = Qwen35_4BTokenizer;
    type Callable = Qwen35LlmCallable;

    const CLI_NAME: &'static str = "qwen3.5-4b";
    const API_NAME: &'static str = "Qwen/Qwen3.5-4B";
    const FAMILY: LlmFamily = LlmFamily::Qwen;

    fn build_prefix_thinking_disabled(
        prompt_before_assistant: &str,
        prompt_after_assistant: &str,
    ) -> String {
        let mut full_prompt = build_simple_qwen_chatml_prefix(prompt_before_assistant, false);
        full_prompt += prompt_after_assistant;
        full_prompt
    }

    fn build_prefix_thinking_enabled(prompt_before_assistant: &str) -> String {
        build_simple_qwen_chatml_prefix(prompt_before_assistant, true)
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
    type Tokenizer = PassthroughTokenizer;
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

    fn callable_from_cli_args(client: Client, _llm_cli_args: &LlmCliArgs) -> Self::Callable {
        GptLlmCallable::new(client)
    }
}

pub struct Gpt5Mini;
impl LlmModelMarker for Gpt5Mini {
    type StringOrTokenArray = String;
    type Tokenizer = PassthroughTokenizer;
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
