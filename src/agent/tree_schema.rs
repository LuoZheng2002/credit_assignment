use crate::{
    agent::tree::Tree,
    datasets::AssetFileDataset,
    direct_answer::generate_raw_answers::LlmModel,
    parallel_process_jsonl::{HasId, read_json},
    version_tracking::{AssetFile, Base64Hash, hash_file},
};
use rusqlite::{Connection, OptionalExtension, Row, Statement, params};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

#[derive(Debug)]
pub struct CompletedTreeStore {
    db_path: PathBuf,
    connection: Connection,
}

impl CompletedTreeStore {
    const CREATE_COMPLETED_TREES_TABLE_SQL: &str = "
        CREATE TABLE IF NOT EXISTS completed_trees (
            id INTEGER PRIMARY KEY,
            payload_json TEXT NOT NULL
        );
    ";

    pub fn new(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        let db_path = db_path.into();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create parent directory for trees sqlite database {}: {}",
                    db_path.display(),
                    e
                )
            })?;
        }
        let connection = Connection::open(&db_path).map_err(|e| {
            format!(
                "Failed to open trees sqlite database {}: {}",
                db_path.display(),
                e
            )
        })?;
        let store = Self {
            db_path,
            connection,
        };
        store.initialize_schema()?;
        Ok(store)
    }

    pub fn initialize_schema(&self) -> Result<(), String> {
        self.connection
            .execute_batch(Self::CREATE_COMPLETED_TREES_TABLE_SQL)
        .map_err(|e| {
            format!(
                "Failed to initialize schema for trees sqlite database {}: {}",
                self.db_path.display(),
                e
            )
        })?;
        Ok(())
    }

    pub fn upsert(&self, tree: &CompletedTree) -> Result<(), String> {
        self.initialize_schema()?;
        let id_i64 = i64::try_from(tree.id).map_err(|_| format!("id out of i64 range: {}", tree.id))?;
        let payload_json = serde_json::to_string(tree).map_err(|e| {
            format!(
                "Failed to serialize completed tree for id={} before sqlite upsert: {}",
                tree.id,
                e
            )
        })?;
        self.connection.execute(
            "
            INSERT INTO completed_trees (id, payload_json)
            VALUES (?1, ?2)
            ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
            ",
            params![id_i64, payload_json],
        )
        .map_err(|e| {
            format!(
                "Failed to upsert completed tree id={} into sqlite database {}: {}",
                tree.id,
                self.db_path.display(),
                e
            )
        })?;
        Ok(())
    }

    pub fn get(&self, id: usize) -> Result<Option<CompletedTree>, String> {
        self.initialize_schema()?;
        let id_i64 = i64::try_from(id).map_err(|_| format!("id out of i64 range: {}", id))?;
        let payload: Option<String> = self
            .connection
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

    #[deprecated(note = "Use two-hop iteration: store.statement()?.try_iter()? instead")]
    pub fn for_each_tree<F>(&self, mut on_tree: F) -> Result<(), String>
    where
        F: FnMut(CompletedTree) -> Result<(), String>,
    {
        let mut tree_scan_statement = self.statement()?;
        let rows = tree_scan_statement.try_iter()?;
        for row in rows {
            let tree = row.map_err(|e| {
                format!(
                    "Failed to read tree row from sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
            on_tree(tree)?;
        }
        Ok(())
    }

    pub fn load_all(&self) -> Result<Vec<CompletedTree>, String> {
        let mut tree_scan_statement = self.statement()?;
        let rows = tree_scan_statement.try_iter()?;
        let mut trees: Vec<CompletedTree> = Vec::new();
        for row in rows {
            let tree = row.map_err(|e| {
                format!(
                    "Failed to read tree row from sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
            trees.push(tree);
        }
        Ok(trees)
    }

    pub fn statement(&self) -> Result<CompletedTreeScanStatement<'_>, String> {
        self.initialize_schema()?;
        let statement = self
            .connection
            .prepare(
                "
                SELECT payload_json
                FROM completed_trees
                ORDER BY id ASC
                ",
            )
            .map_err(|e| {
                format!(
                    "Failed to prepare query for trees sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })?;
        Ok(CompletedTreeScanStatement {
            statement,
            db_path: self.db_path.clone(),
        })
    }
}

pub struct CompletedTreeScanStatement<'conn> {
    statement: Statement<'conn>,
    db_path: PathBuf,
}

impl CompletedTreeScanStatement<'_> {
    pub fn try_iter(
        &mut self,
    ) -> Result<rusqlite::MappedRows<'_, fn(&Row<'_>) -> rusqlite::Result<CompletedTree>>, String>
    {
        self.statement
            .query_map(
                [],
                decode_completed_tree_row as fn(&Row<'_>) -> rusqlite::Result<CompletedTree>,
            )
            .map_err(|e| {
                format!(
                    "Failed to execute query for trees sqlite database {}: {}",
                    self.db_path.display(),
                    e
                )
            })
    }
}

fn decode_completed_tree_row(row: &Row<'_>) -> rusqlite::Result<CompletedTree> {
    let payload_json: String = row.get(0)?;
    serde_json::from_str(&payload_json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
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
        CompletedTreeStore::new(self.file_path()).unwrap()
    }
}
