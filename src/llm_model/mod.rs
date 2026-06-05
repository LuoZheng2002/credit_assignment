pub mod gemma3_4b_it;
pub mod gpt_4o;
pub mod llama31_8b_instruct;
pub mod llm_model_name;
pub mod llm_model_traits;
pub mod qwen25_7b;
pub mod qwen35_08b;
pub mod qwen35_4b;
pub mod qwen3_06b;
pub mod qwen3_4b;
pub mod qwen_shared;

pub use crate::token_array::TokenArray;
pub use crate::token_array::{TokenArrayWithLogprob, TokenLogprobCandidate, Top8Candidates};
pub use gemma3_4b_it::{Gemma3_4BIt, Gemma3_4BItLlmCallable, Gemma3_4BItTokenizer};
pub use gpt_4o::{Gpt4o, Gpt4oLlmCallable, Gpt4oTokenizer};
pub use llama31_8b_instruct::{
    Llama31_8BInstruct, Llama31_8BInstructLlmCallable, Llama31_8BInstructTokenizer,
};
pub use llm_model_name::LlmModelName;
pub(crate) use llm_model_traits::trim_tail_eos_if_needed;
pub use llm_model_traits::{LlmCallable, LlmCliArgs, LlmModelMarker, MyTokenizer};
pub(crate) use qwen3_4b::build_simple_qwen3_chatml_template;
pub use qwen3_4b::{Qwen3_4B, Qwen3_4BLlmCallable, Qwen3_4BTokenizer};
pub use qwen3_06b::{Qwen3_06B, Qwen3_06BLlmCallable, Qwen3_06BTokenizer};
pub use qwen25_7b::{Qwen25_7B, Qwen25LlmCallable, Qwen25Tokenizer};
pub(crate) use qwen35_4b::build_simple_qwen35_chatml_template;
pub use qwen35_4b::{Qwen35_4B, Qwen35_4BLlmCallable, Qwen35_4BTokenizer};
pub use qwen35_08b::{Qwen35_08B, Qwen35_08BLlmCallable, Qwen35_08BTokenizer};
