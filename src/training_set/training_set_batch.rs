use kdtree::{KdTree, distance::squared_euclidean};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::{
    direct_answer::generate_raw_answers::LlmModel,
    em::{em_schema::short_hyperparameter_hash, em_types::EmHyperparameters},
    parallel_process_jsonl::{read_json, write_json},
    sqlite_store::SqliteStore,
    training_set::{
        training_set_formatted::QuestionNodeId,
        training_set_tokenized::{AssetFileTrainingTokenized, TrainingSampleTokenized},
    },
    version_tracking::{AssetFile, Base64Hash, hash_file},
};

#[derive(Debug, Clone)]
pub struct TrainingSampleMeta {
    // pub question_id: usize,
    // pub node_id: usize,
    pub id: QuestionNodeId,
    pub input_length: usize,
    pub advantage: f64,
    pub input_length_normalized: f64,
    pub advantage_normalized: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingBatch(pub Vec<QuestionNodeId>);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssetFileTrainingBatchTracking {
    pub tokenized_hash: Base64Hash,
    pub batch_schema_version: usize,
}

pub type TrainingBatchStore = SqliteStore<usize, TrainingBatch>;

pub struct AssetFileTrainingBatch {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
    pub batch_size: usize,
}

impl AssetFileTrainingBatch {
    const BATCH_SCHEMA_VERSION: usize = 1;

    pub fn hyperparameter_hash(&self) -> String {
        short_hyperparameter_hash(&self.hyperparameters)
    }

    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_training_batch_{}_{}_bs{}.sqlite",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
            self.batch_size,
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_training_batch_{}_{}_bs{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
            self.batch_size,
        )
    }

    pub fn batch_store(&self) -> TrainingBatchStore {
        TrainingBatchStore::new(self.file_path()).unwrap()
    }

    pub fn store_batches(&self, batches: &[TrainingBatch]) {
        let store = self.batch_store();
        store.clear().unwrap();
        for (batch_index, batch) in batches.iter().enumerate() {
            store.upsert(batch_index, batch).unwrap();
        }
    }

    fn normalize(values: &[f64]) -> Vec<f64> {
        assert!(!values.is_empty(), "Cannot normalize an empty value array");
        for value in values {
            assert!(
                value.is_finite(),
                "All values for normalization must be finite"
            );
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f64>()
            / values.len() as f64;
        assert!(
            variance.is_finite() && variance >= 0.0,
            "Normalization variance must be finite and non-negative"
        );
        if variance == 0.0 {
            return vec![0.0; values.len()];
        }
        let std = variance.sqrt();
        values.iter().map(|value| (*value - mean) / std).collect()
    }

    fn sign(value: f64) -> i8 {
        assert!(value.is_finite(), "sign input must be finite");
        if value > 0.0 {
            1
        } else if value < 0.0 {
            -1
        } else {
            0
        }
    }

    pub fn generate_batches_from_tokenized_samples(
        &self,
        tokenized_samples: &[TrainingSampleTokenized],
    ) -> Vec<TrainingBatch> {
        assert!(self.batch_size > 0, "batch_size must be greater than 0");
        if tokenized_samples.is_empty() {
            return Vec::new();
        }

        // to ensure training efficiency, we group training samples with similar absolute advantage values and similar input lengths together.
        // specifically, we first produce TrainingSampleMeta for each sample, and then use a kd tree for querying the nearest neighbors by Euclidean distance
        // For the first sample of each batch, we greedily select the sample with the largest absolute advantage value.
        // Then we keep a record on the cumulative advantage. We select the nearest neighbor to the first sample. If the neighbor's advantage has a different sign from the cumulative advantage, we add it to the batch and remove it from the candidate list.
        // For the consecutive samples, we always compare the distance with the first sample, instead of the last added sample.
        let absolute_advantages: Vec<f64> = tokenized_samples
            .iter()
            .map(|sample| sample.advantage.abs())
            .collect();
        let input_lengths: Vec<f64> = tokenized_samples
            .iter()
            .map(|sample| sample.input_length as f64)
            .collect();
        let advantages_normalized = Self::normalize(&absolute_advantages);
        let input_lengths_normalized = Self::normalize(&input_lengths);

        let sample_meta: Vec<TrainingSampleMeta> = tokenized_samples
            .iter()
            .enumerate()
            .map(|(index, sample)| TrainingSampleMeta {
                id: sample.id,
                input_length: sample.input_length,
                advantage: sample.advantage,
                input_length_normalized: input_lengths_normalized[index],
                advantage_normalized: advantages_normalized[index],
            })
            .collect();

        let mut remaining: HashSet<usize> = (0..tokenized_samples.len()).collect();
        let mut output_batches: Vec<TrainingBatch> = Vec::new();

        while !remaining.is_empty() {
            let seed_index = *remaining
                .iter()
                .max_by(|left, right| {
                    tokenized_samples[**left]
                        .advantage
                        .abs()
                        .total_cmp(&tokenized_samples[**right].advantage.abs())
                })
                .expect("remaining must be non-empty when choosing seed");

            let mut current_batch_indices: Vec<usize> = vec![seed_index];
            remaining.remove(&seed_index);
            let mut cumulative_advantage = tokenized_samples[seed_index].advantage;
            let seed_point = [
                sample_meta[seed_index].advantage_normalized,
                sample_meta[seed_index].input_length_normalized,
            ];

            while current_batch_indices.len() < self.batch_size && !remaining.is_empty() {
                let mut kdtree = KdTree::new(2);
                for candidate_index in &remaining {
                    let candidate_point = [
                        sample_meta[*candidate_index].advantage_normalized,
                        sample_meta[*candidate_index].input_length_normalized,
                    ];
                    kdtree
                        .add(candidate_point, *candidate_index)
                        .expect("Adding a candidate to kd-tree must not fail");
                }

                let nearest_candidates = kdtree
                    .nearest(&seed_point, remaining.len(), &squared_euclidean)
                    .expect("Kd-tree nearest query must not fail");
                assert!(
                    !nearest_candidates.is_empty(),
                    "Nearest query must return candidates when remaining is non-empty"
                );

                let cumulative_sign = Self::sign(cumulative_advantage);
                let chosen_candidate = nearest_candidates
                    .iter()
                    .find_map(|(_, candidate_index)| {
                        let candidate_sign =
                            Self::sign(tokenized_samples[**candidate_index].advantage);
                        if cumulative_sign == 0 {
                            Some(**candidate_index)
                        } else if candidate_sign != 0 && candidate_sign != cumulative_sign {
                            Some(**candidate_index)
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| {
                        *nearest_candidates
                            .first()
                            .expect("Fallback nearest candidate must exist")
                            .1
                    });

                assert!(
                    remaining.remove(&chosen_candidate),
                    "Chosen candidate must still exist in remaining set"
                );
                current_batch_indices.push(chosen_candidate);
                cumulative_advantage += tokenized_samples[chosen_candidate].advantage;
            }

            let batch_ids: Vec<QuestionNodeId> = current_batch_indices
                .iter()
                .map(|index| tokenized_samples[*index].id)
                .collect();
            output_batches.push(TrainingBatch(batch_ids));
        }

        output_batches
    }
}

// use the same style as AssetFileAdvantageComposition
impl AssetFile for AssetFileTrainingBatch {
    type FileModel = TrainingBatchStore;

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        self.batch_store()
    }

    fn synchronize(&self) -> crate::version_tracking::Base64Hash {
        let tokenized_asset = AssetFileTrainingTokenized {
            model: self.model,
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
            hyperparameters: self.hyperparameters.clone(),
        };
        let tokenized_hash = tokenized_asset.synchronize();

        let tracking_content =
            match read_json::<AssetFileTrainingBatchTracking>(self.version_tracking_path()) {
                Ok(mut tracking) => {
                    if tracking.tokenized_hash != tokenized_hash
                        || tracking.batch_schema_version != Self::BATCH_SCHEMA_VERSION
                    {
                        let tokenized_store = tokenized_asset.fetch();
                        let tokenized_samples = tokenized_store.load_all().unwrap();
                        let batches =
                            self.generate_batches_from_tokenized_samples(&tokenized_samples);
                        self.store_batches(&batches);
                        tracking.tokenized_hash = tokenized_hash.clone();
                        tracking.batch_schema_version = Self::BATCH_SCHEMA_VERSION;
                    }
                    tracking
                }
                Err(_) => {
                    let tokenized_store = tokenized_asset.fetch();
                    let tokenized_samples = tokenized_store.load_all().unwrap();
                    let batches = self.generate_batches_from_tokenized_samples(&tokenized_samples);
                    self.store_batches(&batches);
                    AssetFileTrainingBatchTracking {
                        tokenized_hash,
                        batch_schema_version: Self::BATCH_SCHEMA_VERSION,
                    }
                }
            };

        write_json(self.version_tracking_path(), &tracking_content).unwrap();
        hash_file(self.file_path()).unwrap()
    }
}
