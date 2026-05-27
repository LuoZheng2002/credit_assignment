use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::llm_model::{LlmModelMarker, MyTokenizer};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct TokenLogprobCandidate {
    pub token_id: i32,
    pub logprob: f32,
}

pub type Top8Candidates = [TokenLogprobCandidate; 8];

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TokenArrayWithLogprob<M> {
    pub tokens: Vec<i32>,
    pub logprobs: Vec<Top8Candidates>,
    #[serde(skip)]
    _marker: PhantomData<M>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TokenArray<M> {
    pub tokens: Vec<i32>,
    #[serde(skip)]
    _marker: PhantomData<M>,
}

impl<M> TokenArray<M> {
    pub fn from_tokens(tokens: Vec<i32>) -> Self {
        Self {
            tokens,
            _marker: PhantomData,
        }
    }
}

impl<M> Clone for TokenArray<M> {
    fn clone(&self) -> Self {
        Self::from_tokens(self.tokens.clone())
    }
}

impl<M> TokenArrayWithLogprob<M> {
    pub fn from_tokens_and_logprobs(tokens: Vec<i32>, logprobs: Vec<Top8Candidates>) -> Self {
        Self {
            tokens,
            logprobs,
            _marker: PhantomData,
        }
    }
}

impl<M> Clone for TokenArrayWithLogprob<M> {
    fn clone(&self) -> Self {
        Self::from_tokens_and_logprobs(self.tokens.clone(), self.logprobs.clone())
    }
}

impl<M: LlmModelMarker> TokenArray<M> {
    pub fn decode(&self) -> String {
        <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&self.tokens)
    }
}

impl<M: LlmModelMarker> TokenArrayWithLogprob<M> {
    pub fn decode(&self) -> String {
        <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&self.tokens)
    }
}
