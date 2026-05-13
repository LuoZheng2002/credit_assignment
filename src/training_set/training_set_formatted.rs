use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    advantage_composition::{AdvantageCompositionPerTree, AssetFileAdvantageComposition},
    agent::tree_schema::AssetFileTrees,
    direct_answer::generate_raw_answers::LlmModel,
    em::{em_schema::short_hyperparameter_hash, em_types::EmHyperparameters},
    parallel_process_jsonl::{read_json, read_json_lines, write_json, write_jsonl_file},
    training_set::training_set_generation::generate_sample_formatted_from_tree_node,
    version_tracking::{AssetFile, Base64Hash, hash_file},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSampleFormatted {
    pub question_id: usize,
    pub node_id: usize,
    // content_formatted uses mask delimiters:
    // <__start_mask__> ... <__end_mask_with_eos__>
    // and always ends with <__end_mask_with_eos__>.
    pub content_formatted: String,
    pub advantage: f64,
}

pub struct AssetFileTrainingFormatted {
    pub model: LlmModel,
    pub dataset: String,
    pub num_samples: usize,
    pub hyperparameters: EmHyperparameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFileTrainingFormattedTracking {
    pub trees_hash: Base64Hash,
    pub advantage_hash: Base64Hash,
    pub formatted_schema_version: usize,
}

impl AssetFileTrainingFormatted {
    const FORMATTED_SCHEMA_VERSION: usize = 3;

    pub fn hyperparameter_hash(&self) -> String {
        short_hyperparameter_hash(&self.hyperparameters)
    }

    pub fn file_path(&self) -> String {
        format!(
            "results/{}/agent/{}_training_formatted_{}_{}.jsonl",
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
    type FileModel = Vec<TrainingSampleFormatted>;

    fn fetch(&self) -> Self::FileModel {
        self.synchronize();
        read_json_lines(self.file_path()).unwrap()
    }

    fn synchronize(&self) -> crate::version_tracking::Base64Hash {
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
                        write_jsonl_file(self.file_path(), &samples).unwrap();
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
                    write_jsonl_file(self.file_path(), &samples).unwrap();
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
