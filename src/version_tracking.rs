use base64::{Engine, engine::general_purpose};
use serde::{Deserialize, Serialize};

use crate::{datasets::DeepMathQuestion, parallel_process_jsonl::read_json_lines};

pub trait AssetFile {
    type FileModel;
    fn synchronize(&self) -> Base64Hash;
    fn fetch(&self) -> Self::FileModel;
    fn file_path(&self) -> String;
    fn version_tracking_path(&self) -> String;
}

pub struct AssetFileDataset {
    pub dataset: String,
    pub num_samples: usize,
}

impl AssetFile for AssetFileDataset {
    type FileModel = Vec<DeepMathQuestion>;

    // dataset no longer needs to be updated, it only records its hash
    fn synchronize(&self) -> Base64Hash {
        hash_file(self.file_path()).unwrap()
    }
    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        read_json_lines(self.file_path()).unwrap()
    }
    fn file_path(&self) -> String {
        format!(
            "datasets/{}_samples_{}.jsonl",
            self.dataset, self.num_samples
        )
    }
    fn version_tracking_path(&self) -> String {
        unreachable!("Dataset file does not have a tracking file.")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Base64Hash(String);

pub fn hash_file(file_path: impl AsRef<std::path::Path>) -> Result<Base64Hash, String> {
    let file = std::fs::File::open(file_path.as_ref())
        .map_err(|e| format!("Cannot open file {}: {}", file_path.as_ref().display(), e))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut reader, &mut hasher).map_err(|e| {
        format!(
            "Failed to read file {}: {}",
            file_path.as_ref().display(),
            e
        )
    })?;
    let hash = hasher.finalize();
    Ok(Base64Hash(
        general_purpose::STANDARD.encode(hash.as_bytes()),
    ))
}

// #[derive(Clone, Serialize, Deserialize)]
// pub enum AssetFile{
//     Dataset{
//         dataset: String,
//         num_samples: usize,
//     },
//     Trees{
//         model: LlmModel,
//         dataset: String,
//         num_samples: usize,
//     },
//     EmFitPerTree{
//         model: LlmModel,
//         dataset: String,
//         num_samples: usize,
//     },
//     EmFitMeta{
//         model: LlmModel,
//         dataset: String,
//         num_samples: usize,
//     },
//     // we need one source of truth for the per-step advantage contribution
//     // it should imitate the structure of Trees
//     AdvantageComposition {
//         model: LlmModel,
//         dataset: String,
//         num_samples: usize,
//     }
// }

// each asset file's metadata should contain the dependency file, and a method for generating

// can we unify the resume mechanism?
// we can add a "progress" field to the metadata, and it can be "in progress" or "completed".

// if a generated file is marked as "in progress" and its hash changes, the check should have a special return value other than "fresh" or "stale", but "progressed" that makes the downstream file to be able to update a part
// but currently, the most expensive operation is to do the LLM rollout.
// then the next expensive thing is to do the training
// dataset generation should be cheap
// therefore, in this project we do not really need the "progressed" status.
// we have "expensive asset file" and "cheap asset file"

//
