pub const SGLANG_CONTEXT_LENGTH_USE_TOOL: usize = 4096;
pub const SGLANG_CONTEXT_LENGTH_NO_TOOL: usize = 4096;

pub fn sglang_context_length(use_tool: bool) -> usize {
    if use_tool {
        SGLANG_CONTEXT_LENGTH_USE_TOOL
    } else {
        SGLANG_CONTEXT_LENGTH_NO_TOOL
    }
}

pub fn get_max_concurrent_rollout(num_gpus: usize) -> usize {
    200 * num_gpus
}

/// Temperature used for training rollouts (0.7).
pub use crate::fixed_temperatures::TRAINING_TEMPERATURE;

/// Temperature used for validation and testing rollouts (0.0).
pub use crate::fixed_temperatures::VALIDATION_TEMPERATURE;
