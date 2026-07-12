use ordered_float::NotNan;

use crate::direct_tool::hybrid_dataset::DatasetSplit;

/// Temperature used for training rollouts (0.7).
pub fn training_temperature() -> NotNan<f32> {
    NotNan::new(0.7).unwrap()
}

/// Temperature used for validation/testing rollouts (0.0).
pub fn validation_temperature() -> NotNan<f32> {
    NotNan::new(0.0).unwrap()
}

/// Returns the default temperature for a given dataset split.
/// Training uses 0.7; validation and testing use 0.0.
pub fn default_temperature_for_split<S: DatasetSplit>() -> NotNan<f32> {
    if S::IS_TRAINING {
        training_temperature()
    } else {
        validation_temperature()
    }
}
