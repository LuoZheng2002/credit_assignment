use crate::{
    agent::tree::Tree,
    datasets::AssetFileDataset,
    direct_answer::generate_raw_answers::LlmModel,
    parallel_process_jsonl::{HasId, read_json},
    version_tracking::{AssetFile, Base64Hash, hash_file},
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, path::PathBuf};

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

// pub fn get_rollout_trees_path(model: LlmModel, dataset_name: &str, num_samples: usize) -> String {
//     format!(
//         "results/{}/agent/{}_trees_{}.jsonl",
//         model.cli_name(),
//         dataset_name,
//         num_samples
//     )
// }

// // trees and logs should be generated manually instead of from AssetFile

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

#[derive(Debug, Clone)]
pub struct CompletedTreeStore {
    db_path: PathBuf,
    page_size: usize,
}

impl CompletedTreeStore {
    const DEFAULT_PAGE_SIZE: usize = 256;

    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self {
            db_path: db_path.into(),
            page_size: Self::DEFAULT_PAGE_SIZE,
        }
    }

    pub fn with_page_size(db_path: impl Into<PathBuf>, page_size: usize) -> Self {
        assert!(page_size > 0, "page_size must be greater than zero");
        Self {
            db_path: db_path.into(),
            page_size,
        }
    }

    pub fn initialize_schema(&self) -> Result<(), String> {
        let conn = Connection::open(&self.db_path).map_err(|e| {
            format!(
                "Failed to open trees sqlite database {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS completed_trees (
                id INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            ",
        )
        .map_err(|e| {
            format!(
                "Failed to initialize schema for trees sqlite database {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        Ok(())
    }

    pub fn get(&self, id: usize) -> Result<Option<CompletedTree>, String> {
        self.initialize_schema()?;
        let id_i64 = i64::try_from(id).map_err(|_| format!("id out of i64 range: {}", id))?;
        let conn = Connection::open(&self.db_path).map_err(|e| {
            format!(
                "Failed to open trees sqlite database {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        let payload: Option<String> = conn
            .query_row(
                "SELECT payload_json FROM completed_trees WHERE id = ?1",
                params![id_i64],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                format!(
                    "Failed to query completed_trees for id={} from {}: {}",
                    id,
                    self.db_path.display(),
                    e
                )
            })?;
        payload
            .map(|json| {
                serde_json::from_str::<CompletedTree>(&json).map_err(|e| {
                    format!(
                        "Failed to deserialize completed tree payload for id={} from {}: {}",
                        id,
                        self.db_path.display(),
                        e
                    )
                })
            })
            .transpose()
    }

    pub fn iter(&self) -> CompletedTreeIter {
        CompletedTreeIter {
            db_path: self.db_path.clone(),
            page_size: self.page_size,
            current_offset: 0,
            buffered_items: VecDeque::new(),
            exhausted: false,
        }
    }

    pub fn load_all(&self) -> Result<Vec<CompletedTree>, String> {
        self.iter().collect()
    }
}

pub struct CompletedTreeIter {
    db_path: PathBuf,
    page_size: usize,
    current_offset: usize,
    buffered_items: VecDeque<CompletedTree>,
    exhausted: bool,
}

impl CompletedTreeIter {
    fn load_next_page(&mut self) -> Result<(), String> {
        let conn = Connection::open(&self.db_path).map_err(|e| {
            format!(
                "Failed to open trees sqlite database {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS completed_trees (
                id INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            ",
        )
        .map_err(|e| {
            format!(
                "Failed to initialize schema for trees sqlite database {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        let limit_i64 = i64::try_from(self.page_size)
            .map_err(|_| format!("page_size out of i64 range: {}", self.page_size))?;
        let offset_i64 = i64::try_from(self.current_offset)
            .map_err(|_| format!("offset out of i64 range: {}", self.current_offset))?;
        let mut stmt = conn
            .prepare(
                "
                SELECT payload_json
                FROM completed_trees
                ORDER BY id ASC
                LIMIT ?1 OFFSET ?2
                ",
            )
            .map_err(|e| {
                format!(
                    "Failed to prepare paged query for trees sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
        let rows = stmt
            .query_map(params![limit_i64, offset_i64], |row| row.get::<_, String>(0))
            .map_err(|e| {
                format!(
                    "Failed to execute paged query for trees sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
        let mut loaded = VecDeque::new();
        for row in rows {
            let payload = row.map_err(|e| {
                format!(
                    "Failed to read row payload from trees sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
            let tree: CompletedTree = serde_json::from_str(&payload).map_err(|e| {
                format!(
                    "Failed to deserialize row payload from trees sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
            loaded.push_back(tree);
        }
        if loaded.is_empty() {
            self.exhausted = true;
            return Ok(());
        }
        self.current_offset += loaded.len();
        self.buffered_items = loaded;
        Ok(())
    }
}

impl Iterator for CompletedTreeIter {
    type Item = Result<CompletedTree, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.buffered_items.pop_front() {
            return Some(Ok(item));
        }
        if self.exhausted {
            return None;
        }
        match self.load_next_page() {
            Ok(()) => self.buffered_items.pop_front().map(Ok),
            Err(err) => {
                self.exhausted = true;
                Some(Err(err))
            }
        }
    }
}

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
        let store = CompletedTreeStore::new(self.file_path());
        store.initialize_schema().unwrap();
        store
    }
}
