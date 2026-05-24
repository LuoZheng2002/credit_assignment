use crate::token_array::{TokenArrayWithLogprob, TokenLogprobCandidate};

pub mod proto {
    tonic::include_proto!("vllm_wrapper.v1");
}

#[derive(Clone, Debug)]
pub enum VllmPrompt {
    Text(String),
    TokenIds(Vec<i32>),
}

#[derive(Clone, Debug)]
pub struct VllmRequest {
    pub model_name: String,
    pub prompt: VllmPrompt,
    pub max_tokens: usize,
    pub include_stop_str_in_output: bool,
    pub requires_logprobs: bool,
    pub stop: Vec<String>,
    pub temperature: f32,
}

#[derive(Clone, Debug, Copy)]
pub struct VllmLogprob {
    pub token_id: i32,
    pub logprob: f32,
}

#[derive(Clone, Debug)]
pub struct VllmLogprobs {
    pub tokens: Vec<i32>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<[VllmLogprob; 8]>,
}

#[derive(Clone, Debug)]
pub enum VllmResponse {
    Success {
        response_text: String,
        logprobs: Option<VllmLogprobs>,
    },
    Error {
        error_message: String,
    },
}

impl VllmRequest {
    pub fn to_proto(&self) -> proto::VllmRequest {
        let prompt = Some(proto::VllmPrompt {
            prompt: Some(match &self.prompt {
                VllmPrompt::Text(text) => proto::vllm_prompt::Prompt::Text(text.clone()),
                VllmPrompt::TokenIds(token_ids) => {
                    proto::vllm_prompt::Prompt::TokenIds(proto::TokenIds {
                        token_ids: token_ids.clone(),
                    })
                }
            }),
        });

        let max_tokens = u32::try_from(self.max_tokens).expect("max_tokens must fit in u32");
        proto::VllmRequest {
            model_name: self.model_name.clone(),
            prompt,
            max_tokens,
            include_stop_str_in_output: self.include_stop_str_in_output,
            requires_logprobs: self.requires_logprobs,
            stop: self.stop.clone(),
            temperature: self.temperature,
        }
    }
}

impl VllmResponse {
    pub fn from_proto(response: proto::VllmResponse) -> Self {
        match response.response {
            Some(proto::vllm_response::Response::Success(success)) => {
                let logprobs = success.logprobs.map(|logprobs| {
                    let proto::VllmLogprobs {
                        tokens,
                        token_logprobs,
                        top_logprobs,
                    } = logprobs;

                    let mut mapped_top_logprobs = Vec::with_capacity(top_logprobs.len());
                    for (idx, top) in top_logprobs.into_iter().enumerate() {
                        let mut top8 = [VllmLogprob {
                            token_id: *tokens.get(idx).unwrap_or(&0),
                            logprob: f32::NEG_INFINITY,
                        }; 8];
                        for (slot, candidate) in top.candidates.into_iter().take(8).enumerate() {
                            top8[slot] = VllmLogprob {
                                token_id: candidate.token_id,
                                logprob: candidate.logprob,
                            };
                        }
                        mapped_top_logprobs.push(top8);
                    }

                    VllmLogprobs {
                        tokens,
                        token_logprobs,
                        top_logprobs: mapped_top_logprobs,
                    }
                });

                VllmResponse::Success {
                    response_text: success.response_text,
                    logprobs,
                }
            }
            Some(proto::vllm_response::Response::Error(error)) => VllmResponse::Error {
                error_message: error.error_message,
            },
            None => VllmResponse::Error {
                error_message: "vllm wrapper response missing payload".to_string(),
            },
        }
    }

    pub fn into_token_array_with_logprob(self) -> Result<TokenArrayWithLogprob, String> {
        match self {
            VllmResponse::Error { error_message } => Err(error_message),
            VllmResponse::Success {
                response_text,
                logprobs,
            } => {
                let Some(logprobs) = logprobs else {
                    return Err(
                        "vllm wrapper returned success without logprobs for logprob request"
                            .to_string(),
                    );
                };

                if logprobs.tokens.len() != logprobs.top_logprobs.len() {
                    return Err(format!(
                        "vllm wrapper logprobs length mismatch: tokens={} top_logprobs={}",
                        logprobs.tokens.len(),
                        logprobs.top_logprobs.len(),
                    ));
                }

                let VllmLogprobs {
                    tokens,
                    token_logprobs: _,
                    top_logprobs,
                } = logprobs;

                let mapped_logprobs = top_logprobs
                    .into_iter()
                    .map(|candidates| {
                        candidates.map(|candidate| TokenLogprobCandidate {
                            token_id: candidate.token_id,
                            logprob: candidate.logprob,
                        })
                    })
                    .collect();

                Ok(TokenArrayWithLogprob {
                    tokens,
                    decoded_string: response_text,
                    logprobs: mapped_logprobs,
                })
            }
        }
    }
}
