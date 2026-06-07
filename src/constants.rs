use crate::llm_model::LlmModelMarker;

pub const SGLANG_CONTEXT_LENGTH_COMMON: usize = 8192;
pub const SGLANG_CONTEXT_LENGTH_GEMMA: usize = 4096;

pub fn sglang_context_length_for_model<M: LlmModelMarker>() -> usize {
    if M::CLI_NAME.starts_with("gemma-") {
        SGLANG_CONTEXT_LENGTH_GEMMA
    } else {
        SGLANG_CONTEXT_LENGTH_COMMON
    }
}
