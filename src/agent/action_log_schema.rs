use serde::{Deserialize, Serialize};

use crate::{
    agent::{
        single_dataset::AssetFileSingleDataset, tree_action_log::TreeActionLog,
        tree_reconstruction::reconstruct_completed_tree, tree_schema::CompletedTree,
    },
    asset_file::{AssetFile, Base64Hash},
    json_line_util::{read_json, write_json},
    llm_model::LlmModelName,
    sqlite_store::SqliteStore,
    util::block_on_async,
};

pub type TreeActionLogStore = SqliteStore<usize, TreeActionLog>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileActionLogsTracking {
    pub dataset_hash: Base64Hash,
    pub action_log_schema_version: usize,
}

pub struct AssetFileActionLogs {
    pub model: LlmModelName,
    pub dataset: String,
    pub num_samples: usize,
}

impl AssetFileActionLogs {
    pub const ACTION_LOG_SCHEMA_VERSION: usize = 1;

    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_rollout_log_{}.sqlite",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_rollout_log_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples
        )
    }

    pub async fn open_store(&self) -> TreeActionLogStore {
        let dataset_hash = AssetFileSingleDataset {
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
        }
        .synchronize();

        match read_json::<AssetFileActionLogsTracking>(self.version_tracking_path()) {
            Ok(tracking) => {
                assert_eq!(
                    tracking.action_log_schema_version,
                    Self::ACTION_LOG_SCHEMA_VERSION,
                    "Action log schema version mismatch for {}",
                    self.file_path()
                );
                assert_eq!(
                    tracking.dataset_hash,
                    dataset_hash,
                    "Action log dataset hash mismatch for {}",
                    self.file_path()
                );
            }
            Err(_) => {
                let tracking = AssetFileActionLogsTracking {
                    dataset_hash,
                    action_log_schema_version: Self::ACTION_LOG_SCHEMA_VERSION,
                };
                write_json(self.version_tracking_path(), &tracking).unwrap();
            }
        }

        TreeActionLogStore::initialize_if_missing(self.file_path()).await
    }

    pub fn load_all_logs_sync(&self) -> Vec<TreeActionLog> {
        block_on_async(async {
            let store = self.open_store().await;
            store.load_all().await.unwrap()
        })
    }

    pub fn load_completed_trees_sync(&self) -> Vec<CompletedTree> {
        let logs = self.load_all_logs_sync();
        let mut trees = Vec::new();
        let mut incomplete_count = 0usize;
        for log in logs {
            let completed_tree = reconstruct_completed_tree(&log);
            if completed_tree.trajectory.completed {
                trees.push(completed_tree);
            } else {
                incomplete_count += 1;
            }
        }
        if incomplete_count > 0 {
            println!(
                "[Warning] Excluding {} incomplete action logs from completed tree view",
                incomplete_count
            );
        }
        trees
    }
}
