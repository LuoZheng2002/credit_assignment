use serde::{Deserialize, Serialize};

use crate::{
    direct_answer::generate_raw_answers::LlmModel, em::em_types::EmHyperparameters, training_set::training_set_tokenized::TrainingSampleTokenized, version_tracking::AssetFile
};

#[derive(Debug, Clone)]
pub struct TrainingSampleMeta {
    pub question_id: usize,
    pub node_id: usize,
    pub input_length: usize,
    pub advantage: f64,
    pub input_length_normalized: f64,
    pub advantage_normalized: f64,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingBatch(Vec<TrainingSampleTokenized>);

pub struct AssetFileTrainingBatch {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
    pub batch_size: usize,
}

impl AssetFileTrainingBatch {
    pub fn generate_batches_from_tokenized_samples(&self, tokenized_samples: &[TrainingSampleTokenized]) -> Vec<TrainingBatch> {
        // to ensure training efficiency, we group training samples with similar absolute advantage values and similar input lengths together.
        // specifically, we first produce TrainingSampleMeta for each sample, and then use a kd tree for querying the nearest neighbors by Euclidean distance
        // For the first sample of each batch, we greedily select the sample with the largest absolute advantage value.
        // Then we keep a record on the cumulative advantage. We select the nearest neighbor to the first sample. If the neighbor's advantage has a different sign from the cumulative advantage, we add it to the batch and remove it from the candidate list.
        // For the consecutive samples, we always compare the distance with the first sample, instead of the last added sample.
        todo!()
    }
}

// use the same style as AssetFileAdvantageComposition
impl AssetFile for AssetFileTrainingBatch {
    type FileModel = Vec<TrainingBatch>;
    fn fetch(&self) -> Self::FileModel {
        todo!()
    }
    fn synchronize(&self) -> crate::version_tracking::Base64Hash {
        // use generate_batches_from_tokenized_samples
        todo!()
    }
}
