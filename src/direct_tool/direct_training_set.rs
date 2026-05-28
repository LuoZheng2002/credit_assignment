use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    sync::Arc,
};

use ordered_float::NotNan;
use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    sqlite_store::{SqliteBusyRetryConfig, SqliteStore},
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::{DirectTree, SegmentContent, SegmentId},
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        hybrid_dataset::HybridDatasetQuestion,
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::{read_json, write_json},
    llm_model::LlmModelMarker,
};

pub struct TrajectoryHeapItem<M: LlmModelMarker> {
    pub trajectory: DirectTrainingTrajectory<M>,
    pub average_absolute_advantage: NotNan<f32>,
}

impl<M: LlmModelMarker> PartialEq for TrajectoryHeapItem<M> {
    fn eq(&self, other: &Self) -> bool {
        self.average_absolute_advantage == other.average_absolute_advantage
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
        self.average_absolute_advantage
            .cmp(&other.average_absolute_advantage)
    }
}

pub async fn rollout_logs_to_training_trajectories<M: LlmModelMarker>(
    action_log_store: SqliteStore<usize, DirectTreeActionLog<M>>,
    max_num_training_trajectories: usize,
    statistics_file_path: String,
) -> Vec<DirectTrainingTrajectory<M>> {
    if max_num_training_trajectories == 0 {
        let statistics = DirectTrainingSetStatistics {
            average_absolute_advantages_sorted: Vec::new(),
            max_average_absolute_advantage: 0.0,
            min_average_absolute_advantage: 0.0,
            average_absolute_advantage_cutoff: 0.0,
            total_trajectories: 0,
            adopted_trajectories: 0,
        };
        write_json(statistics_file_path.clone(), &statistics).unwrap();
        println!(
            "max_average_absolute_advantage: {}",
            statistics.max_average_absolute_advantage
        );
        println!(
            "min_average_absolute_advantage: {}",
            statistics.min_average_absolute_advantage
        );
        println!(
            "average_absolute_advantage_cutoff: {}",
            statistics.average_absolute_advantage_cutoff
        );
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
    // we need a min heap to collect the top n trajectories with the highest
    // average absolute segment advantage across all action logs, where n is
    // the number of trajectories we want to train on in total.
    let mut min_heap: BinaryHeap<Reverse<TrajectoryHeapItem<M>>> = BinaryHeap::new();
    let mut all_average_absolute_advantages: Vec<f32> = Vec::new();
    let mut total_trajectories = 0_usize;
    let mut join_set = JoinSet::new();
    let mut pending_trajectories_by_index: BTreeMap<usize, Vec<DirectTrainingTrajectory<M>>> =
        BTreeMap::new();
    let mut next_index_to_reduce = 0usize;
    let task_semaphore = Arc::new(Semaphore::new(30));

    for (index, key) in keys.iter().enumerate() {
        if index % log_interval == 0 {
            println!("Processing action logs: {}/{}", index, num_keys);
        }
        let action_log = action_log_store.get(*key).await.unwrap().unwrap();
        let task_permit = task_semaphore.clone().acquire_owned().await.unwrap();
        join_set.spawn_blocking(move || {
            let _task_permit = task_permit;
            (index, action_log_to_candidate_trajectories::<M>(action_log))
        });

        while let Some(result) = join_set.try_join_next() {
            let (finished_index, finished_trajectories) =
                result.expect("action_log_to_candidate_trajectories task panicked");
            pending_trajectories_by_index.insert(finished_index, finished_trajectories);

            while let Some(trajectories) =
                pending_trajectories_by_index.remove(&next_index_to_reduce)
            {
                fold_candidate_trajectories(
                    trajectories,
                    &mut total_trajectories,
                    &mut all_average_absolute_advantages,
                    &mut min_heap,
                    max_num_training_trajectories,
                );
                next_index_to_reduce += 1;
            }
        }
    }

    while let Some(result) = join_set.join_next().await {
        let (finished_index, finished_trajectories) =
            result.expect("action_log_to_candidate_trajectories task panicked");
        pending_trajectories_by_index.insert(finished_index, finished_trajectories);

        while let Some(trajectories) = pending_trajectories_by_index.remove(&next_index_to_reduce) {
            fold_candidate_trajectories(
                trajectories,
                &mut total_trajectories,
                &mut all_average_absolute_advantages,
                &mut min_heap,
                max_num_training_trajectories,
            );
            next_index_to_reduce += 1;
        }
    }

    let kept_items: Vec<TrajectoryHeapItem<M>> = min_heap.into_iter().map(|item| item.0).collect();
    let average_absolute_advantage_cutoff = kept_items
        .iter()
        .map(|item| item.trajectory.average_absolute_segment_advantage)
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

    all_average_absolute_advantages.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let adopted_trajectories = kept_trajectories.len();
    let max_average_absolute_advantage = *all_average_absolute_advantages.first().unwrap_or(&0.0);
    let min_average_absolute_advantage = *all_average_absolute_advantages.last().unwrap_or(&0.0);
    // we want to output a histogram and the advantage cutoff
    // and total samples and adopted samples

    let statistics = DirectTrainingSetStatistics {
        average_absolute_advantages_sorted: all_average_absolute_advantages,
        max_average_absolute_advantage,
        min_average_absolute_advantage,
        average_absolute_advantage_cutoff,
        total_trajectories,
        adopted_trajectories,
    };
    write_json(statistics_file_path.clone(), &statistics).unwrap();
    println!(
        "max_average_absolute_advantage: {}",
        statistics.max_average_absolute_advantage
    );
    println!(
        "min_average_absolute_advantage: {}",
        statistics.min_average_absolute_advantage
    );
    println!(
        "average_absolute_advantage_cutoff: {}",
        statistics.average_absolute_advantage_cutoff
    );
    println!("total_trajectories: {}", statistics.total_trajectories);
    println!("adopted_trajectories: {}", statistics.adopted_trajectories);
    kept_trajectories
}

fn fold_candidate_trajectories<M: LlmModelMarker>(
    candidate_trajectories: Vec<DirectTrainingTrajectory<M>>,
    total_trajectories: &mut usize,
    all_average_absolute_advantages: &mut Vec<f32>,
    min_heap: &mut BinaryHeap<Reverse<TrajectoryHeapItem<M>>>,
    max_num_training_trajectories: usize,
) {
    for trajectory in candidate_trajectories {
        *total_trajectories += 1;
        let average_absolute_advantage = NotNan::new(trajectory.average_absolute_segment_advantage)
            .expect("Average absolute segment advantage must not be NaN");
        all_average_absolute_advantages.push(*average_absolute_advantage);

        min_heap.push(Reverse(TrajectoryHeapItem {
            trajectory,
            average_absolute_advantage,
        }));
        if min_heap.len() > max_num_training_trajectories {
            min_heap.pop();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectTrainingSetStatistics {
    pub average_absolute_advantages_sorted: Vec<f32>, // sorted from high to low
    pub max_average_absolute_advantage: f32,
    pub min_average_absolute_advantage: f32,
    pub average_absolute_advantage_cutoff: f32,
    pub total_trajectories: usize,
    pub adopted_trajectories: usize,
}

fn action_log_to_candidate_trajectories<M: LlmModelMarker>(
    action_log: DirectTreeActionLog<M>,
) -> Vec<DirectTrainingTrajectory<M>> {
    let tree = DirectTree::<M>::from_action_log(&action_log);
    if !tree.completed {
        return Vec::new();
    }
    let root_segment_id = tree
        .root_segment_id
        .expect("DirectTree must have root_segment_id");
    let mut segment_advantages = tree.calculate_segment_advantages(None);
    for segment_id in tree.segments.keys().copied() {
        segment_advantages.entry(segment_id).or_insert(0.0);
    }
    let mut trajectories: Vec<DirectTrainingTrajectory<M>> = Vec::new();
    let mut leaf_segment_ids: BTreeSet<SegmentId> =
        tree.leaf_segment_judgments.keys().cloned().collect();
    while !leaf_segment_ids.is_empty() {
        let mut leaf_to_average_absolute_advantage = BTreeMap::new();
        for leaf in leaf_segment_ids.iter() {
            let segment_ids = tree.get_trajectory_segments_till_id(*leaf);
            let non_root_segment_count = segment_ids
                .iter()
                .filter(|&&id| id != root_segment_id)
                .count();
            let average_absolute_advantage = segment_ids
                .iter()
                .filter(|&&id| id != root_segment_id)
                .map(|id| segment_advantages.get(id).unwrap().abs())
                .sum::<f32>()
                / non_root_segment_count.max(1) as f32;
            leaf_to_average_absolute_advantage.insert(*leaf, average_absolute_advantage);
        }
        let (best_leaf, best_average_absolute_advantage) = leaf_to_average_absolute_advantage
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let segment_ids = tree.get_trajectory_segments_till_id(best_leaf);
        let mut input_ids: Vec<i32> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        let mut advantages: Vec<f32> = Vec::new();
        let mut sum_absolute_advantage = 0.0;
        let mut non_root_segment_count = 0usize;
        for segment_id in segment_ids.iter() {
            let segment = tree.segments.get(segment_id).unwrap();
            let segment_advantage = segment_advantages.get_mut(segment_id).unwrap();
            if *segment_id != root_segment_id {
                sum_absolute_advantage += segment_advantage.abs();
                non_root_segment_count += 1;
            }
            for content in segment.content.iter() {
                match content {
                    SegmentContent::Prompt(token_array)
                    | SegmentContent::ToolResponse(token_array) => {
                        input_ids.extend(token_array.tokens.iter());
                        labels.extend(vec![-100; token_array.tokens.len()]); // we set the labels for the prompt tokens to -100 so that they will be ignored in the loss calculation
                        advantages.extend(vec![*segment_advantage; token_array.tokens.len()]); // we assign the same advantage to all tokens in the segment
                    }
                    SegmentContent::ReasoningOrToolCall {
                        tokens,
                        complete: _,
                    } => {
                        input_ids.extend(tokens.tokens.iter());
                        labels.extend(tokens.tokens.iter());
                        advantages.extend(vec![*segment_advantage; tokens.tokens.len()]);
                    }
                }
            }
            *segment_advantage = 0.0; // we set the advantage of the taken segments to 0
        }
        let average_absolute_advantage =
            sum_absolute_advantage / non_root_segment_count.max(1) as f32;
        assert_eq!(average_absolute_advantage, best_average_absolute_advantage);
        trajectories.push(DirectTrainingTrajectory {
            question: tree.question.clone(),
            input_ids,
            labels,
            advantages,
            average_absolute_segment_advantage: average_absolute_advantage,
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
    pub labels: Vec<i32>, // we may not need to let model learn to stop at tool-call boundaries or end since our framework already handled this
    pub advantages: Vec<f32>,
    pub average_absolute_segment_advantage: f32,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<M>,
}

pub struct AssetFileTrainingTrajectories<M: LlmModelMarker> {
    // pub model: LlmModelName,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize,
    pub max_num_training_trajectories: usize,
    pub _phantom: std::marker::PhantomData<M>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetFileTrainingTrajectoriesTracking {
    pub rollout_log_hash: Base64Hash,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize,
    pub max_num_training_trajectories: usize,
}

impl<M: LlmModelMarker> AssetFileTrainingTrajectories<M> {
    fn to_short_hash(&self) -> String {
        let serialized = serde_json::to_vec(&(
            &&self.config_nickname,
            &self.rollout_config,
            &self.posterior_calculation_config,
            &self.epoch,
        ))
        .unwrap();
        let hash = blake3::hash(&serialized);
        let short_hash = hex::encode(&hash.as_bytes()[..4]); // Take the first 4 bytes for a shorter hash
        short_hash
    }
    pub fn file_path(&self) -> String {
        format!(
            "results/{}/{}/epoch_{}/training_trajectories_{}.sqlite",
            M::CLI_NAME,
            self.config_nickname,
            self.epoch,
            self.to_short_hash()
        )
    }
    pub fn statistics_file_path(&self) -> String {
        format!(
            "results/{}/{}/epoch_{}/training_trajectories_{}_statistics.json",
            M::CLI_NAME,
            self.config_nickname,
            self.epoch,
            self.to_short_hash()
        )
    }
    fn version_tracking_path(&self) -> String {
        format!(
            "results_version_tracking/{}/{}/epoch_{}/training_trajectories_{}_tracking.json",
            M::CLI_NAME,
            self.config_nickname,
            self.epoch,
            self.to_short_hash()
        )
    }
}

#[async_trait::async_trait]
impl<M: LlmModelMarker> AssetFile for AssetFileTrainingTrajectories<M> {
    type FileModel = SqliteStore<usize, DirectTrainingTrajectory<M>>;
    async fn fetch(&self) -> Self::FileModel {
        self.synchronize().await;
        SqliteStore::<usize, DirectTrainingTrajectory<M>>::assume_initialized(self.file_path(), 1)
            .await
    }
    async fn synchronize(&self) -> Base64Hash {
        let asset_file_rollout_logs = AssetFileDirectTreeActionLogs::<M> {
            nickname: self.config_nickname.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch: self.epoch,
            _phantom: std::marker::PhantomData,
        };
        let rollout_log_hash = asset_file_rollout_logs.synchronize().await;
        let new_tracking_content = AssetFileTrainingTrajectoriesTracking {
            rollout_log_hash,
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch: self.epoch,
            max_num_training_trajectories: self.max_num_training_trajectories,
        };
        let stale = if let Ok(tracking_content) =
            read_json::<AssetFileTrainingTrajectoriesTracking>(self.version_tracking_path())
        {
            let mut is_stale = false;
            if tracking_content != new_tracking_content {
                is_stale = true;
            }
            // check if file exists
            if !std::path::Path::new(&self.file_path()).exists() {
                is_stale = true;
            }
            is_stale
        } else {
            true
        };
        if stale {
            println!("Training trajectories file is stale or does not exist. Regenerating...",);
            // if so, we first delete the target file
            if std::path::Path::new(&self.file_path()).exists() {
                std::fs::remove_file(&self.file_path()).unwrap();
            }
            // initialize database
            let db =
                SqliteStore::<usize, DirectTrainingTrajectory<M>>::initialize(self.file_path(), 1)
                    .await;
            let rollout_logs = asset_file_rollout_logs.fetch().await;
            let training_trajectories = rollout_logs_to_training_trajectories::<M>(
                rollout_logs,
                self.max_num_training_trajectories,
                self.statistics_file_path(),
            )
            .await;
            for (i, trajectory) in training_trajectories.into_iter().enumerate() {
                db.upsert(i, &trajectory, SqliteBusyRetryConfig::none())
                    .await
                    .unwrap();
            }
            println!("Finished generating training trajectories file.");
        }
        write_json(self.version_tracking_path(), &new_tracking_content).unwrap();
        hash_file(self.file_path()).expect("Failed to hash training trajectories file")
    }
}
