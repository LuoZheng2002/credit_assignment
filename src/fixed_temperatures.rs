use ordered_float::NotNan;

use crate::hybrid_dataset::DatasetSplit;

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
