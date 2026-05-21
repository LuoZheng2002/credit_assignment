use crate::{
    agent::action_log_schema::AssetFileActionLogs,
    agent::tree::Tree,
    asset_file::{AssetFile, Base64Hash, hash_file},
    json_line_util::HasId,
    llm_model::LlmModelName,
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

pub struct AssetFileTrees {
    pub model: LlmModelName,
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
    rows: Vec<CompletedTree>,
}

impl CompletedTreeStore {
    pub fn from_rows(mut rows: Vec<CompletedTree>) -> Self {
        rows.sort_by_key(|row| row.id);
        Self { rows }
    }

    pub fn load_all(&self) -> Result<Vec<CompletedTree>, String> {
        Ok(self.rows.clone())
    }

    pub fn statement(&self) -> Result<CompletedTreeStatement, String> {
        Ok(CompletedTreeStatement {
            rows: self.rows.clone(),
        })
    }
}

pub struct CompletedTreeStatement {
    rows: Vec<CompletedTree>,
}

impl CompletedTreeStatement {
    pub fn try_iter(
        &mut self,
    ) -> Result<std::vec::IntoIter<Result<CompletedTree, String>>, String> {
        Ok(self
            .rows
            .clone()
            .into_iter()
            .map(Ok)
            .collect::<Vec<_>>()
            .into_iter())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileTreesTracking {
    pub dataset_hash: Base64Hash,
}
impl AssetFile for AssetFileTrees {
    type FileModel = CompletedTreeStore;

    fn synchronize(&self) -> Base64Hash {
        let action_logs = AssetFileActionLogs {
            model: self.model,
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
        };
        hash_file(action_logs.file_path()).unwrap()
    }

    fn fetch(&self) -> Self::FileModel {
        let action_logs = AssetFileActionLogs {
            model: self.model,
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
        };
        CompletedTreeStore::from_rows(action_logs.load_completed_trees_sync())
    }
}
