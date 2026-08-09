use ordered_float::NotNan;

use crate::hybrid_dataset::DatasetSplit;

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
    300 * num_gpus
}

/// Temperature used for training rollouts (0.7).
pub const TRAINING_TEMPERATURE: f32 = 0.7;

/// Temperature used for validation and testing rollouts (0.0).
pub const VALIDATION_TEMPERATURE: f32 = 0.0;

/// Returns the fixed temperature for a given dataset split.
/// Training uses 0.7, validation/testing uses 0.0.
pub fn temperature_by_split<S: DatasetSplit>() -> NotNan<f32> {
    NotNan::new(if S::IS_TRAINING {
        TRAINING_TEMPERATURE
    } else {
        VALIDATION_TEMPERATURE
    })
    .unwrap()
}
