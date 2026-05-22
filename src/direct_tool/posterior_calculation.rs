// In em_schema.rs, there is a EmFitMeta
// we may not have a meta
// we are considering if the mapping from temperature to accuracy should be counted as meta
// it is like an experiment setting; maybe we can put it in the tracking file

use std::collections::BTreeMap;

use ordered_float::NotNan;
use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    sqlite_store::SqliteStore,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureToAccuracy(pub BTreeMap<NotNan<f32>, f32>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFilePosteriorFitTracking {
    pub action_log_hash: Base64Hash,
    pub temperature_to_accuracy: TemperatureToAccuracy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFilePosteriorFit {
    pub model: String,
    pub temperature_to_accuracy: TemperatureToAccuracy,
}

pub fn short_temperature_hash(temperature_to_accuracy: &TemperatureToAccuracy) -> String {
    let serialized_mapping = serde_json::to_vec(temperature_to_accuracy)
        .expect("Failed to serialize temperature to accuracy mapping");
    let hash = blake3::hash(&serialized_mapping);
    let short_hash = hex::encode(&hash.as_bytes()[..2]); // take the first 2 bytes for a shorter hash
    assert_eq!(short_hash.len(), 4); // 2 bytes should give us 4 hex characters
    short_hash
}

impl AssetFilePosteriorFit {
    fn temperature_hash(&self) -> String {
        short_temperature_hash(&self.temperature_to_accuracy)
    }
    fn file_path(&self) -> String {
        format!(
            "results/{}/direct_tool/posterior_fit_{}.sqlite",
            self.model,
            self.temperature_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/direct_tool/posterior_fit_{}.version.json",
            self.model,
            self.temperature_hash()
        )
    }
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PosteriorFitPerTree {}

#[async_trait::async_trait]
impl AssetFile for AssetFilePosteriorFit {
    type FileModel = SqliteStore<usize, PosteriorFitPerTree>;
    async fn synchronize(&self) -> Base64Hash {
        hash_file(self.file_path()).unwrap()
    }

    async fn fetch(&self) -> Self::FileModel {
        // self.synchronize().await;
        // let tracking = research_utility::json_line_util::read_json::<AssetFilePosteriorFitTracking>(self.version_tracking_path()).unwrap();
        // assert_eq!(tracking.action_log_hash, self.synchronize().await);
        // self.clone()
        todo!()
    }
}

// normalize within tree, and clipping once

// avoid penalizing correct steps
