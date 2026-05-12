use crate::{direct_answer::generate_raw_answers::LlmModel, em::em_types::EmHyperparameters, version_tracking::AssetFile};

pub struct TrainingSampleFormatted {
    pub question_id: usize,
    pub node_id: usize,
    pub content_formatted: String,
    pub advantage: f64,
}


pub struct AssetFileTrainingFormatted {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
}

impl AssetFileTrainingFormatted {

}

// use the same style as AssetFileAdvantageComposition
impl AssetFile for AssetFileTrainingFormatted{
    type FileModel = Vec<TrainingSampleFormatted>;
    fn fetch(&self) -> Self::FileModel {
        todo!()
    }
    fn synchronize(&self) -> crate::version_tracking::Base64Hash {
        // use generate_sample_formatted_from_tree_node
        todo!()
    }
}