// In em_schema.rs, there is a EmFitMeta
// we may not have a meta
// we are considering if the mapping from temperature to accuracy should be counted as meta
// it is like an experiment setting; maybe we can put it in the tracking file

use std::{collections::BTreeMap, path::PathBuf};

use clap::ValueEnum;
use ordered_float::NotNan;
use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    sqlite_store::SqliteStore,
};
use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        direct_tree_action_log::{
            AssetFileDirectTreeActionLogs, DirectRolloutConfig, DirectTreeActionLog,
        },
        posterior_calculation::action_log_to_posterior_fit,
    },
    json_line_util::{read_json, write_json},
    llm_model::{LlmModelMarker, LlmModelName},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PosteriorCalculationConfig {
    pub temperature_to_accuracy: BTreeMap<NotNan<f32>, NotNan<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFilePosteriorFitTracking {
    pub action_log_hash: Base64Hash,
    pub direct_rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AssetFilePosteriorFit<M: LlmModelMarker> {
    // pub model: LlmModelName,
    pub direct_rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub _phantom: std::marker::PhantomData<M>,
}

impl<M: LlmModelMarker> AssetFilePosteriorFit<M> {
    fn short_hash(&self) -> String {
        let serialized = serde_json::to_vec(&(
            &self.direct_rollout_config,
            &self.posterior_calculation_config,
        ))
        .expect("Failed to serialize direct rollout config and posterior calculation config");
        let hash = blake3::hash(&serialized);
        let short_hash = hex::encode(&hash.as_bytes()[..4]); // take the first 4 bytes for a shorter hash
        assert_eq!(short_hash.len(), 8); // 4 bytes should give us 8 hex characters
        short_hash
    }
    fn file_path(&self) -> String {
        format!(
            "results/{}/direct_tool/posterior_fit_{}.sqlite",
            M::CLI_NAME,
            self.short_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/direct_tool/posterior_fit_{}.version.json",
            M::CLI_NAME,
            self.short_hash()
        )
    }
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PosteriorFitPerTree {}

#[async_trait::async_trait]
impl<M: LlmModelMarker> AssetFile for AssetFilePosteriorFit<M> {
    type FileModel = SqliteStore<usize, PosteriorFitPerTree>;
    async fn synchronize(&self) -> Base64Hash {
        // synchronize all dependency assets
        let asset_file_direct_tree_action_logs = AssetFileDirectTreeActionLogs {
            model: LlmModelName::from_str(M::CLI_NAME, true).unwrap(),
            config: self.direct_rollout_config.clone(),
        };
        let action_log_hash = asset_file_direct_tree_action_logs.synchronize().await;
        // tracking file does not exist is the same as tracking file is stale
        let stale = if let Ok(tracking_content) =
            read_json::<AssetFilePosteriorFitTracking>(self.version_tracking_path())
        {
            let mut is_stale = false;
            if tracking_content.action_log_hash != action_log_hash {
                println!(
                    "Action log hash mismatch: expected {:?}, got {:?}",
                    tracking_content.action_log_hash, action_log_hash
                );
                is_stale = true;
            }
            if tracking_content.direct_rollout_config != self.direct_rollout_config {
                println!("Direct rollout config mismatch between tracking and current config");
                is_stale = true;
            }
            if tracking_content.posterior_calculation_config != self.posterior_calculation_config {
                println!(
                    "Posterior calculation config mismatch between tracking and current config"
                );
                is_stale = true;
            }
            is_stale
        } else {
            true
        };
        // if stale, we regenerate
        if stale {
            let rollout_store = asset_file_direct_tree_action_logs.fetch().await;
            calculate_posterior_and_store::<M>(
                rollout_store,
                self.file_path(),
                &self.posterior_calculation_config,
            )
            .await;
        }
        // after that, we rewrite tracking even if not stale
        let tracking_content = AssetFilePosteriorFitTracking {
            action_log_hash,
            direct_rollout_config: self.direct_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
        };
        // write the tracking file
        write_json(self.version_tracking_path(), &tracking_content).unwrap();
        hash_file(self.file_path()).unwrap()
    }

    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::assume_initialized(self.file_path()).await
    }
}

pub async fn calculate_posterior_and_store<M: LlmModelMarker>(
    rollout_store: SqliteStore<usize, DirectTreeActionLog>,
    posterior_path: impl Into<PathBuf>,
    posterior_calculation_config: &PosteriorCalculationConfig,
) {
    let posterior_path = posterior_path.into();
    // remove the file at posterior_path if it exists
    if posterior_path.exists() {
        std::fs::remove_file(&posterior_path).expect("Failed to remove existing posterior file");
    }
    let posterior_store =
        SqliteStore::<usize, PosteriorFitPerTree>::initialize(posterior_path).await;
    let rollout_keys = rollout_store
        .get_keys()
        .await
        .expect("Failed to get keys from rollout store");
    for key in rollout_keys {
        let action_log = rollout_store
            .get(key)
            .await
            .expect("Failed to get action log from rollout store");
        let action_log = action_log.expect("Corresponding row does not exist");
        // calculate posterior fit for the tree corresponding to this action log
        let posterior_fit = action_log_to_posterior_fit::<M>(&action_log, posterior_calculation_config);
        posterior_store
            .upsert(key, &posterior_fit)
            .await
            .expect("Failed to insert posterior fit into posterior store");
    }
}

// normalize within tree, and clipping once

// avoid penalizing correct steps
