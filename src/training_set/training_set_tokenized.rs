use serde::{Deserialize, Serialize};

use crate::{
    direct_answer::generate_raw_answers::LlmModel, em::em_types::EmHyperparameters,
    training_set::training_set_formatted::TrainingSampleFormatted, version_tracking::AssetFile,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSampleTokenized {
    pub question_id: usize,
    pub node_id: usize,
    pub input_ids: Vec<usize>,
    pub labels: Vec<usize>,
    pub input_length: usize,
    pub advantage: f64,
}

pub struct AssetFileTrainingTokenized {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
}

impl AssetFileTrainingTokenized {
    pub fn formatted_to_tokenized(
        &self,
        formatted_sample: &TrainingSampleFormatted,
    ) -> TrainingSampleTokenized {
        todo!()
    }
    pub fn generate_tokenized_samples(
        &self,
        formatted_samples: &[TrainingSampleFormatted],
    ) -> Vec<TrainingSampleTokenized> {
        formatted_samples
            .iter()
            .map(|sample| self.formatted_to_tokenized(sample))
            .collect()
    }
}

// use the same style as AssetFileAdvantageComposition
impl AssetFile for AssetFileTrainingTokenized {
    type FileModel = Vec<TrainingSampleTokenized>;
    fn fetch(&self) -> Self::FileModel {
        todo!()
    }
    fn synchronize(&self) -> crate::version_tracking::Base64Hash {
        // use generate_tokenized_samples
        todo!()
    }
}
