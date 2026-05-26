use std::{cmp::Reverse, collections::{BTreeMap, BTreeSet, BinaryHeap}};

use clap::ValueEnum;
use ordered_float::NotNan;
use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    sqlite_store::SqliteStore,
};
use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::{DirectTree, SegmentContent, SegmentId},
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        hybrid_dataset::HybridDatasetQuestion,
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::{read_json, write_json},
    llm_model::{LlmModelMarker, LlmModelName},
};

pub struct TrajectoryHeapItem<M: LlmModelMarker> {
    pub trajectory: DirectTrainingTrajectory<M>,
    pub average_advantage: NotNan<f32>,
}

impl<M: LlmModelMarker> PartialEq for TrajectoryHeapItem<M> {
    fn eq(&self, other: &Self) -> bool {
        self.average_advantage == other.average_advantage
    }
}

impl<M: LlmModelMarker> Eq for TrajectoryHeapItem<M> {}

impl<M: LlmModelMarker> PartialOrd for TrajectoryHeapItem<M> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<M: LlmModelMarker> Ord for TrajectoryHeapItem<M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.average_advantage.cmp(&other.average_advantage)
    }
}

pub async fn rollout_logs_to_training_trajectories<M: LlmModelMarker>(
    action_log_store: SqliteStore<usize, DirectTreeActionLog>,
    max_num_training_trajectories: usize,
    statistics_file_path: String,
) -> Vec<DirectTrainingTrajectory<M>> {
    if max_num_training_trajectories == 0 {
        let statistics = DirectTrainingSetStatistics {
            average_advantages_sorted: Vec::new(),
            max_average_advantage: 0.0,
            min_average_advantage: 0.0,
            average_advantage_cutoff: 0.0,
            total_trajectories: 0,
            adopted_trajectories: 0,
        };
        write_json(statistics_file_path.clone(), &statistics).unwrap();
        println!("max_average_advantage: {}", statistics.max_average_advantage);
        println!("min_average_advantage: {}", statistics.min_average_advantage);
        println!("average_advantage_cutoff: {}", statistics.average_advantage_cutoff);
        println!("total_trajectories: {}", statistics.total_trajectories);
        println!("adopted_trajectories: {}", statistics.adopted_trajectories);
        return Vec::new();
    }

    // iterate through all action logs
    let mut keys = action_log_store.get_keys().await.unwrap();
    let num_keys = keys.len();
    let log_interval = std::cmp::max(num_keys / 200, 1);
    // we want to make a histogram

    keys.sort(); // ensure deterministic order
    // we need a min heap to collect the top n trajectories with the highest average segment advantage across all action logs, where n is the number of trajectories we want to train on in total.
    let mut min_heap: BinaryHeap<Reverse<TrajectoryHeapItem<M>>> = BinaryHeap::new();
    let mut all_average_advantages: Vec<f32> = Vec::new();
    let mut total_trajectories = 0_usize;
    for (index, key) in keys.iter().enumerate() {
        if index % log_interval == 0 {
            println!("Processing action logs: {}/{}", index, num_keys);
        }
        let action_log = action_log_store.get(*key).await.unwrap().unwrap();
        let candidate_trajectories = action_log_to_candidate_trajectories::<M>(action_log);
        for trajectory in candidate_trajectories {
            total_trajectories += 1;
            let average_advantage = NotNan::new(trajectory.average_segment_advantage)
                .expect("Average segment advantage must not be NaN");
            all_average_advantages.push(*average_advantage);

            min_heap.push(Reverse(TrajectoryHeapItem {
                trajectory,
                average_advantage,
            }));
            if min_heap.len() > max_num_training_trajectories {
                min_heap.pop();
            }
        }
    }

    let kept_items: Vec<TrajectoryHeapItem<M>> = min_heap.into_iter().map(|item| item.0).collect();
    let average_advantage_cutoff = kept_items
        .iter()
        .map(|item| item.trajectory.average_segment_advantage)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let mut kept_trajectories: Vec<DirectTrainingTrajectory<M>> =
        kept_items.into_iter().map(|item| item.trajectory).collect();

    for trajectory in kept_trajectories.iter() {
        assert_eq!(
            trajectory.input_ids.len(),
            trajectory.labels.len(),
            "kept trajectory must satisfy input_ids.len() == labels.len(); question_flat_id={}",
            trajectory.question.flat_id
        );
        assert_eq!(
            trajectory.input_ids.len(),
            trajectory.advantages.len(),
            "kept trajectory must satisfy input_ids.len() == advantages.len(); question_flat_id={}",
            trajectory.question.flat_id
        );
    }

    kept_trajectories.sort_by_key(|trajectory| trajectory.input_ids.len());

    all_average_advantages.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let adopted_trajectories = kept_trajectories.len();
    let max_average_advantage = *all_average_advantages.first().unwrap_or(&0.0);
    let min_average_advantage = *all_average_advantages.last().unwrap_or(&0.0);
    // we want to output a histogram and the advantage cutoff
    // and total samples and adopted samples

    let statistics = DirectTrainingSetStatistics {
        average_advantages_sorted: all_average_advantages,
        max_average_advantage,
        min_average_advantage,
        average_advantage_cutoff,
        total_trajectories,
        adopted_trajectories,
    };
    write_json(statistics_file_path.clone(), &statistics).unwrap();
    println!("max_average_advantage: {}", statistics.max_average_advantage);
    println!("min_average_advantage: {}", statistics.min_average_advantage);
    println!("average_advantage_cutoff: {}", statistics.average_advantage_cutoff);
    println!("total_trajectories: {}", statistics.total_trajectories);
    println!("adopted_trajectories: {}", statistics.adopted_trajectories);
    kept_trajectories
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectTrainingSetStatistics {
    pub average_advantages_sorted: Vec<f32>, // sorted from high to low
    pub max_average_advantage: f32,
    pub min_average_advantage: f32,
    pub average_advantage_cutoff: f32,
    pub total_trajectories: usize,
    pub adopted_trajectories: usize,
}

fn action_log_to_candidate_trajectories<M: LlmModelMarker>(
    action_log: DirectTreeActionLog,
) -> Vec<DirectTrainingTrajectory<M>> {
    let tree = DirectTree::<M>::from_action_log(&action_log);
    let mut segment_advantages = tree.calculate_segment_advantages(None);
    for segment_id in tree.segments.keys().copied() {
        segment_advantages.entry(segment_id).or_insert(0.0);
    }
    let mut trajectories: Vec<DirectTrainingTrajectory<M>> = Vec::new();
    let mut leaf_segment_ids: BTreeSet<SegmentId> =
        tree.leaf_segment_judgments.keys().cloned().collect();
    while !leaf_segment_ids.is_empty() {
        let mut leaf_to_average_advantage = BTreeMap::new();
        for leaf in leaf_segment_ids.iter() {
            let segment_ids = tree.get_trajectory_segments_till_id(*leaf);
            let average_advantage = segment_ids
                .iter()
                .map(|id| segment_advantages.get(id).unwrap())
                .sum::<f32>()
                / segment_ids.len() as f32;
            leaf_to_average_advantage.insert(*leaf, average_advantage);
        }
        let (best_leaf, best_average_advantage) = leaf_to_average_advantage
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let segment_ids = tree.get_trajectory_segments_till_id(best_leaf);
        let mut input_ids: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut advantages: Vec<f32> = Vec::new();
        let mut sum_advantage = 0.0;
        for segment_id in segment_ids.iter() {
            let segment = tree.segments.get(segment_id).unwrap();
            let segment_advantage = segment_advantages.get_mut(segment_id).unwrap();
            sum_advantage += *segment_advantage;
            for content in segment.content.iter() {
                match content {
                    SegmentContent::Prompt(token_array)
                    | SegmentContent::ToolResponse(token_array) => {
                        input_ids.extend(token_array.tokens.iter());
                        labels.extend(vec![-100; token_array.tokens.len()]); // we set the labels for the prompt tokens to -100 so that they will be ignored in the loss calculation
                        advantages.extend(vec![*segment_advantage; token_array.tokens.len()]); // we assign the same advantage to all tokens in the segment
                    }
                    SegmentContent::ReasoningOrToolCall { tokens, complete: _ } => {
                        input_ids.extend(tokens.tokens.iter());
                        labels.extend(tokens.tokens.iter());
                        advantages.extend(vec![*segment_advantage; tokens.tokens.len()]);
                    }
                }
            }
            *segment_advantage = 0.0; // we set the advantage of the taken segments to 0
        }
        let average_advantage = sum_advantage / segment_ids.len() as f32;
        assert_eq!(average_advantage, best_average_advantage);
        trajectories.push(DirectTrainingTrajectory {
            question: tree.question.clone(),
            input_ids,
            labels,
            advantages,
            average_segment_advantage: average_advantage,
            _phantom: std::marker::PhantomData::<M>,
        });
        leaf_segment_ids.remove(&best_leaf);
    }
    trajectories
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectTrainingTrajectory<M: LlmModelMarker> {
    pub question: HybridDatasetQuestion,
    pub input_ids: Vec<i32>,
    pub labels: Vec<i32>, // we may not need to let model learn to stop after tool_wait or end since our framework already handled this
    pub advantages: Vec<f32>,
    pub average_segment_advantage: f32,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<M>,
}

pub struct AssetFileTrainingTrajectories<M: LlmModelMarker> {
    // pub model: LlmModelName,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub max_num_training_trajectories: usize,
    pub _phantom: std::marker::PhantomData<M>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetFileTrainingTrajectoriesTracking {
    pub rollout_log_hash: Base64Hash,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub max_num_training_trajectories: usize,
}

impl<M: LlmModelMarker> AssetFileTrainingTrajectories<M> {
    fn to_short_hash(&self) -> String {
        let serialized = serde_json::to_vec(&(
            &&self.config_nickname,
            &self.rollout_config,
            &self.posterior_calculation_config,
        ))
        .unwrap();
        let hash = blake3::hash(&serialized);
        let short_hash = hex::encode(&hash.as_bytes()[..4]); // Take the first 4 bytes for a shorter hash
        short_hash
    }
    fn model_name(&self) -> LlmModelName {
        LlmModelName::from_str(M::CLI_NAME, true).unwrap()
    }
    pub fn file_path(&self) -> String {
        format!(
            "results/{}/training_trajectories_{}_{}.sqlite",
            M::CLI_NAME,
            self.config_nickname,
            self.to_short_hash()
        )
    }
    pub fn statistics_file_path(&self) -> String {
        format!(
            "results/{}/training_trajectories_{}_{}_statistics.json",
            M::CLI_NAME,
            self.config_nickname,
            self.to_short_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/training_trajectories_{}_{}_tracking.json",
            M::CLI_NAME,
            self.config_nickname,
            self.to_short_hash()
        )
    }
}

#[async_trait::async_trait]
impl<M: LlmModelMarker> AssetFile for AssetFileTrainingTrajectories<M> {
    type FileModel = SqliteStore<usize, DirectTrainingTrajectory<M>>;
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::<usize, DirectTrainingTrajectory<M>>::assume_initialized(self.file_path())
            .await
    }
    async fn synchronize(&self) -> Base64Hash {
        let asset_file_rollout_logs = AssetFileDirectTreeActionLogs {
            model: self.model_name(),
            nickname: self.config_nickname.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
        };
        let rollout_log_hash = asset_file_rollout_logs.synchronize().await;
        let new_tracking_content = AssetFileTrainingTrajectoriesTracking {
            rollout_log_hash,
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            max_num_training_trajectories: self.max_num_training_trajectories,
        };
        let stale = if let Ok(tracking_content) =
            read_json::<AssetFileTrainingTrajectoriesTracking>(self.version_tracking_path())
        {
            tracking_content != new_tracking_content
        } else {
            true
        };
        if stale {
            println!(
                "Training trajectories file is stale or does not exist. Regenerating...",
            );
            // if so, we first delete the target file
            if std::path::Path::new(&self.file_path()).exists() {
                std::fs::remove_file(&self.file_path()).unwrap();
            }
            // initialize database
            let db =
                SqliteStore::<usize, DirectTrainingTrajectory<M>>::initialize(self.file_path())
                    .await;
            let rollout_logs = asset_file_rollout_logs.fetch().await;
            let training_trajectories = rollout_logs_to_training_trajectories::<M>(
                rollout_logs,
                self.max_num_training_trajectories,
                self.statistics_file_path(),
            )
            .await;
            for (i, trajectory) in training_trajectories.into_iter().enumerate() {
                db.upsert(i, &trajectory).await.unwrap();
            }
            println!("Finished generating training trajectories file.");
        }
        write_json(self.version_tracking_path(), &new_tracking_content).unwrap();
        hash_file(self.file_path()).expect("Failed to hash training trajectories file")
    }
}
