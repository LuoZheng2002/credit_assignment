use crate::{
    agent::tree::Tree,
    datasets::AssetFileDataset,
    direct_answer::generate_raw_answers::LlmModel,
    parallel_process_jsonl::{HasId, read_json},
    sqlite_store::SqliteStore,
    version_tracking::{AssetFile, Base64Hash, hash_file},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StepQualityRatio {
    pub tool_numerator: usize,
    pub tool_denominator: usize,
    pub complete_numerator: usize,
    pub complete_denominator: usize,
    pub focused_numerator: usize,
    pub focused_denominator: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CountRatio {
    pub numerator: usize,
    pub denominator: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompletedTree {
    pub id: usize,
    pub correct_answer: String,
    pub step_quality_ratio: StepQualityRatio,
    pub failed_and_aborted_ratio: CountRatio,
    pub trajectory: Tree,
    pub question: String,
}

impl HasId for CompletedTree {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_rollout_log_path(model: LlmModel, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/agent/{}_rollout_log_{}.jsonl",
        model.cli_name(),
        dataset_name,
        num_samples
    )
}

pub struct AssetFileTrees {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
}

impl AssetFileTrees {
    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_trees_{}.sqlite",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_trees_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }
}

// Kept as a type alias for compatibility at call sites.
pub type CompletedTreeStore = SqliteStore<usize, CompletedTree>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileTreesTracking {
    pub dataset_hash: Base64Hash,
}
impl AssetFile for AssetFileTrees {
    type FileModel = CompletedTreeStore;
    fn synchronize(&self) -> Base64Hash {
        let dataset = AssetFileDataset {
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
        };
        let dataset_hash = dataset.synchronize();
        // if tracking exists then the target file also exists
        let tracking_file = match read_json::<AssetFileTreesTracking>(&self.version_tracking_path())
        {
            Ok(tracking) => {
                if tracking.dataset_hash != dataset_hash {
                    // tracking.dataset_hash = dataset_hash.clone();
                    // normally we need to regenerate the stale file, but since this file is expensive to generate, we only print a warning.
                    println!(
                        "[Warning]: The dependency of trees file {} has changed (dataset content changed: {})",
                        self.file_path(),
                        dataset.file_path(),
                    );
                }
                tracking
            }
            Err(_) => {
                panic!(
                    "Tracking file for trees file {} does not exist.",
                    self.file_path()
                );
            }
        };
        std::fs::write(
            self.version_tracking_path(),
            serde_json::to_string_pretty(&tracking_file).unwrap(),
        )
        .unwrap();
        hash_file(self.file_path()).unwrap()
    }
    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        CompletedTreeStore::new(self.file_path()).unwrap()
    }
}
