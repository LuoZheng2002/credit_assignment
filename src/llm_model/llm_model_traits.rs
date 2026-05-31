use async_trait::async_trait;
use clap::Args;
use reqwest::Client;

use crate::{llm_model::TokenArrayWithLogprob, token_array::TokenArray};

#[derive(Args, Clone, Debug)]
pub struct LlmCliArgs {
    #[arg(long)]
    pub sglang_port: Option<u16>, // used when the model is served by sglang
}

pub trait LlmModelMarker: Sized + Send + Sync + 'static {
    type Tokenizer: MyTokenizer<Self>;
    type Callable: LlmCallable<Self> + Send + Sync + 'static;

    const CLI_NAME: &'static str;
    const API_NAME: &'static str;
}

pub trait MyTokenizer<M: LlmModelMarker>: Send + Sync + 'static {
    fn tokenize(prompt: String) -> TokenArray<M>;
    fn apply_chat_template_and_tokenize(prompt: String, enable_thinking: bool) -> TokenArray<M>;
    fn apply_python_response_template_and_tokenize(raw_python_response: String) -> TokenArray<M>;
    fn encode_to_i32_ids(text: &str) -> Vec<i32>;
    fn decode_i32_ids(token_ids: &[i32]) -> String;
    fn eos_token_id() -> i32;
}

#[async_trait]
pub trait LlmCallable<M: LlmModelMarker>: Clone + Send + Sync {
    fn from_cli_args(client: Client, llm_cli_args: &LlmCliArgs) -> Self
    where
        Self: Sized;
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
