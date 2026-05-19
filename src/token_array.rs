use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct TokenLogprobCandidate {
    pub token_id: i32,
    pub logprob: f32,
}

pub type Top8Candidates = [TokenLogprobCandidate; 8];

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenArrayWithLogprob {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
    pub logprobs: Vec<Top8Candidates>,
    pub logprobs_past_end: Top8Candidates, // the logprobs of the next token after the end of the content
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenArray {
    pub tokens: Vec<i32>,
    pub decoded_string: String,
}
