use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    agent::advantage_composition::{AdvantageCompositionPerTree, AssetFileAdvantageComposition},
    agent::tree_schema::AssetFileTrees,
    asset_file::{AssetFile, Base64Hash, hash_file},
    em::{em_schema::short_hyperparameter_hash, em_types::EmHyperparameters},
    json_line_util::{read_json, write_json},
    llm_model_name::LlmModelName,
    sqlite_store::{SqliteStore, SqliteStoreKey},
    training_set::training_set_generation::generate_sample_formatted_from_tree_node,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct QuestionNodeId {
    pub question_id: usize,
    pub node_id: usize,
}
impl SqliteStoreKey for QuestionNodeId {
    fn to_key_text(&self) -> String {
        format!("q{}_n{}", self.question_id, self.node_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSampleFormatted {
    pub id: QuestionNodeId,
    // content_formatted uses mask delimiters:
    // <__start_mask__> ... <__end_mask_with_eos__>
    // and always ends with <__end_mask_with_eos__>.
    pub content_formatted: String,
    pub advantage: f64,
}

pub struct AssetFileTrainingFormatted {
    pub model: LlmModelName,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
}

pub type TrainingSampleFormattedStore = SqliteStore<QuestionNodeId, TrainingSampleFormatted>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileTrainingFormattedTracking {
    pub trees_hash: Base64Hash,
    pub advantage_hash: Base64Hash,
    pub formatted_schema_version: usize,
}

impl AssetFileTrainingFormatted {
    const FORMATTED_SCHEMA_VERSION: usize = 4;

    pub fn hyperparameter_hash(&self) -> String {
        short_hyperparameter_hash(&self.hyperparameters)
    }

    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_training_formatted_{}_{}.sqlite",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    pub fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/agent/{}_training_formatted_{}_{}.version.json",
            self.model.cli_name(),
            self.dataset,
            self.num_samples,
            self.hyperparameter_hash(),
        )
    }

    pub fn sample_store(&self) -> TrainingSampleFormattedStore {
        TrainingSampleFormattedStore::new(self.file_path()).unwrap()
    }

    pub fn store_formatted_samples(&self, samples: &[TrainingSampleFormatted]) {
        let store = self.sample_store();
        store.clear().unwrap();
        let mut seen_ids: BTreeSet<QuestionNodeId> = BTreeSet::new();
        for sample in samples {
            assert!(
                seen_ids.insert(sample.id),
                "Duplicate QuestionNodeId found in formatted samples: {:?}",
                sample.id
            );
            store.upsert(sample.id, sample).unwrap();
        }
    }

    fn generate_formatted_samples(
        &self,
        trees: &crate::agent::tree_schema::CompletedTreeStore,
        advantage_per_tree: &[AdvantageCompositionPerTree],
    ) -> Vec<TrainingSampleFormatted> {
        let advantage_by_id: BTreeMap<usize, &AdvantageCompositionPerTree> = advantage_per_tree
            .iter()
            .map(|tree| (tree.question_id, tree))
            .collect();
        assert_eq!(
            advantage_by_id.len(),
            advantage_per_tree.len(),
            "Duplicate question_id found in advantage_per_tree"
        );

        let mut output: Vec<TrainingSampleFormatted> = Vec::new();
        let mut seen_tree_ids: BTreeSet<usize> = BTreeSet::new();
        let mut tree_scan_statement = trees.statement().unwrap();
        let rows = tree_scan_statement.try_iter().unwrap();
        for row in rows {
            let tree = row.unwrap();
            assert!(
                seen_tree_ids.insert(tree.id),
                "Duplicate tree id found in trees: {}",
                tree.id
            );
            let advantage = advantage_by_id.get(&tree.id).unwrap();
            for node in &advantage.per_node {
                output.push(generate_sample_formatted_from_tree_node(
                    &tree,
                    advantage,
                    node.node_id,
                    self.model,
                ));
            }
        }
        let advantage_tree_ids: BTreeSet<usize> = advantage_by_id.keys().copied().collect();
        assert_eq!(
            seen_tree_ids, advantage_tree_ids,
            "Tree id set must match advantage tree id set"
        );
        output
    }
}

// use the same style as AssetFileAdvantageComposition
impl AssetFile for AssetFileTrainingFormatted {
    type FileModel = TrainingSampleFormattedStore;

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        self.sample_store()
    }

    fn synchronize(&self) -> crate::asset_file::Base64Hash {
        let asset_file_trees = AssetFileTrees {
            model: self.model,
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
        };
        let trees_hash = asset_file_trees.synchronize();

        let asset_file_advantage = AssetFileAdvantageComposition {
            model: self.model,
            dataset: self.dataset.clone(),
            num_samples: self.num_samples,
            hyperparameters: self.hyperparameters.clone(),
        };
        let advantage_hash = asset_file_advantage.synchronize();

        let tracking_content =
            match read_json::<AssetFileTrainingFormattedTracking>(self.version_tracking_path()) {
                Ok(mut tracking) => {
                    if tracking.trees_hash != trees_hash
                        || tracking.advantage_hash != advantage_hash
                        || tracking.formatted_schema_version != Self::FORMATTED_SCHEMA_VERSION
                    {
                        let trees = asset_file_trees.fetch();
                        let advantage_per_tree = asset_file_advantage.fetch();
                        let samples = self.generate_formatted_samples(&trees, &advantage_per_tree);
                        self.store_formatted_samples(&samples);
                        tracking.trees_hash = trees_hash.clone();
                        tracking.advantage_hash = advantage_hash.clone();
                        tracking.formatted_schema_version = Self::FORMATTED_SCHEMA_VERSION;
                    }
                    tracking
                }
                Err(_) => {
                    let trees = asset_file_trees.fetch();
                    let advantage_per_tree = asset_file_advantage.fetch();
                    let samples = self.generate_formatted_samples(&trees, &advantage_per_tree);
                    self.store_formatted_samples(&samples);
                    AssetFileTrainingFormattedTracking {
                        trees_hash,
                        advantage_hash,
                        formatted_schema_version: Self::FORMATTED_SCHEMA_VERSION,
                    }
                }
            };

        write_json(self.version_tracking_path(), &tracking_content).unwrap();
        hash_file(self.file_path()).unwrap()
    }
}
