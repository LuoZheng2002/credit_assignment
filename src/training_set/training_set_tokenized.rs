use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{
    direct_answer::generate_raw_answers::LlmModel, em::em_types::EmHyperparameters,
    em::em_schema::short_hyperparameter_hash,
    parallel_process_jsonl::{read_json, read_json_lines, write_json, write_jsonl_file},
    training_set::training_set_formatted::{AssetFileTrainingFormatted, TrainingSampleFormatted},
    version_tracking::{AssetFile, Base64Hash, hash_file},
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileTrainingTokenizedTracking {
    pub formatted_hash: Base64Hash,
}

impl AssetFileTrainingTokenized {
    pub fn hyperparameter_hash(&self) -> String {
        short_hyperparameter_hash(&self.hyperparameters)
    }

    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_training_tokenized_{}_{}.jsonl",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_training_tokenized_{}_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    fn load_tokenizer(&self) -> Tokenizer {
        assert!(self.model.is_qwen(), "Training tokenization currently supports Qwen models only");
        Tokenizer::from_pretrained(self.model.api_name(), None).unwrap_or_else(|err| {
            panic!(
                "Failed to load tokenizer for model {} from {}: {}",
                self.model.cli_name(),
                self.model.api_name(),
                err
            )
        })
    }

    fn formatted_to_tokenized_with_tokenizer(
        &self,
        formatted_sample: &TrainingSampleFormatted,
        tokenizer: &Tokenizer,
    ) -> TrainingSampleTokenized {
        let encoding = tokenizer
            .encode(formatted_sample.content_formatted.clone(), false)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to tokenize sample (question_id={}, node_id={}): {}",
                    formatted_sample.question_id, formatted_sample.node_id, err
                )
            });
        let input_ids: Vec<usize> = encoding.get_ids().iter().map(|id| *id as usize).collect();
        assert!(!input_ids.is_empty(), "Tokenized input must be non-empty");

        TrainingSampleTokenized {
            question_id: formatted_sample.question_id,
            node_id: formatted_sample.node_id,
            labels: input_ids.clone(),
            input_length: input_ids.len(),
            input_ids,
            advantage: formatted_sample.advantage,
        }
    }

    pub fn formatted_to_tokenized(
        &self,
        formatted_sample: &TrainingSampleFormatted,
    ) -> TrainingSampleTokenized {
        let tokenizer = self.load_tokenizer();
        self.formatted_to_tokenized_with_tokenizer(formatted_sample, &tokenizer)
    }

    pub fn generate_tokenized_samples(
        &self,
        formatted_samples: &[TrainingSampleFormatted],
    ) -> Vec<TrainingSampleTokenized> {
        let tokenizer = self.load_tokenizer();
        formatted_samples
            .iter()
            .map(|sample| self.formatted_to_tokenized_with_tokenizer(sample, &tokenizer))
            .collect()
    }
}

// use the same style as AssetFileAdvantageComposition
impl AssetFile for AssetFileTrainingTokenized {
    type FileModel = Vec<TrainingSampleTokenized>;

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        read_json_lines(self.file_path()).unwrap()
    }

    fn synchronize(&self) -> crate::version_tracking::Base64Hash {
        let asset_file_training_formatted = AssetFileTrainingFormatted {
            model: self.model,
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
            hyperparameters: self.hyperparameters.clone(),
        };
        let formatted_hash = asset_file_training_formatted.synchronize();

        let tracking_content = match read_json::<AssetFileTrainingTokenizedTracking>(
            self.version_tracking_path(),
        ) {
            Ok(mut tracking) => {
                if tracking.formatted_hash != formatted_hash {
                    let formatted_samples = asset_file_training_formatted.fetch();
                    let tokenized_samples = self.generate_tokenized_samples(&formatted_samples);
                    write_jsonl_file(self.file_path(), &tokenized_samples).unwrap_or_else(|err| {
                        panic!(
                            "Failed to write tokenized training set output to {}: {}",
                            self.file_path(),
                            err
                        )
                    });
                    tracking.formatted_hash = formatted_hash.clone();
                }
                tracking
            }
            Err(_) => {
                let formatted_samples = asset_file_training_formatted.fetch();
                let tokenized_samples = self.generate_tokenized_samples(&formatted_samples);
                write_jsonl_file(self.file_path(), &tokenized_samples).unwrap_or_else(|err| {
                    panic!(
                        "Failed to write tokenized training set output to {}: {}",
                        self.file_path(),
                        err
                    )
                });
                AssetFileTrainingTokenizedTracking { formatted_hash }
            }
        };
        write_json(self.version_tracking_path(), &tracking_content).unwrap();
        hash_file(self.file_path()).unwrap()
    }
}
