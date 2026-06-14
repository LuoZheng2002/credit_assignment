use async_trait::async_trait;
use reqwest::Client;

use crate::{llm_model::TokenArrayWithLogprob, token_array::TokenArray};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferenceEndpoint {
    SglangPort(u16),
    SglangBaseUrl(String),
}

impl InferenceEndpoint {
    pub fn from_cli_options(
        sglang_port: Option<u16>,
        sglang_base_url: Option<String>,
    ) -> Result<Self, String> {
        match (sglang_port, sglang_base_url) {
            (Some(port), None) => Ok(Self::SglangPort(port)),
            (None, Some(base_url)) => Ok(Self::SglangBaseUrl(base_url)),
            (Some(_), Some(_)) => {
                Err("--sglang-port and --sglang-base-url cannot both be set".to_string())
            }
            (None, None) => {
                Err("either --sglang-port or --sglang-base-url must be provided".to_string())
            }
        }
    }
}

pub trait LlmModelMarker: Sized + Send + Sync + 'static + Clone {
    type Tokenizer: MyTokenizer<Self>;
    type Callable: LlmCallable<Self> + Send + Sync + 'static;

    const CLI_NAME: &'static str;
    const API_NAME: &'static str;
    const MODEL_LABEL: &'static str;
}

pub trait MyTokenizer<M: LlmModelMarker>: Send + Sync + 'static {
    fn tokenize(prompt: String) -> TokenArray<M>;
    fn apply_chat_template_and_tokenize(prompt: String, enable_thinking: bool) -> TokenArray<M>;
    fn apply_python_response_template_and_tokenize(
        raw_python_response: String,
        enable_thinking: bool,
    ) -> TokenArray<M>;
    fn encode_to_i32_ids(text: &str) -> Vec<i32>;
    fn decode_i32_ids(token_ids: &[i32]) -> String;
    fn eos_token_id() -> i32;
}

#[async_trait]
pub trait LlmCallable<M: LlmModelMarker>: Clone + Send + Sync {
    fn from_inference_endpoint(client: Client, inference_endpoint: &InferenceEndpoint) -> Self
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
    ) -> Result<TokenArrayWithLogprob<M>, String>;
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
