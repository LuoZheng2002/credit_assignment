pub const SGLANG_CONTEXT_LENGTH_USE_TOOL: usize = 4096;
pub const SGLANG_CONTEXT_LENGTH_NO_TOOL: usize = 2048;

pub fn sglang_context_length(use_tool: bool) -> usize {
    if use_tool {
        SGLANG_CONTEXT_LENGTH_USE_TOOL
    } else {
        SGLANG_CONTEXT_LENGTH_NO_TOOL
    }
}
