use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

use ordered_float::NotNan;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use research_utility::{
    asset_file::{AssetFile, Base64Hash, hash_file},
    log_message::{log_info, log_key_value_pair, log_master_progress, log_warning},
    sqlite_store::{SqliteBusyRetryConfig, SqliteStore},
};
use serde::{Deserialize, Serialize};

use crate::{
    direct_tool::{
        direct_rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
        direct_tree::{DirectTree, SegmentContent, SegmentId},
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        hybrid_dataset::{DatasetSplit, HybridDatasetQuestion},
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
    advantage_calculation_policy: AdvantageCalculationPolicy,
) -> Vec<DirectTrainingTrajectory<M>> {
    if max_num_training_trajectories == 0 {
        log_warning("Max num training trajectory is 0");
        return Vec::new();
    }

    // iterate through all action logs
    let mut keys = action_log_store.get_keys().await.unwrap();
    let num_keys = keys.len();
    // we want to make a histogram

    keys.sort(); // ensure deterministic order
    // we need a min heap to collect the top n trajectories with the highest
    // average absolute segment advantage across all action logs, where n is
    // the number of trajectories we want to train on in total.
    let mut min_heap: BinaryHeap<Reverse<TrajectoryHeapItem<M>>> = BinaryHeap::new();
    let mut all_average_absolute_advantages: Vec<f32> = Vec::new();
    let mut total_trajectories = 0_usize;
    let mut finished_samples = 0usize;
    let nproc = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let num_worker_threads = std::cmp::max((nproc + 1) / 2, 1);
    let worker_pool = ThreadPoolBuilder::new()
        .num_threads(num_worker_threads)
        .build()
        .expect("Failed to build rayon worker pool for training trajectory conversion");
    log_info(format!("Converting action logs to trajectories with rayon threads: {}", num_worker_threads));
    let chunk_size = std::cmp::max(num_worker_threads * 4, 1);
    let mut pending_chunk: Vec<(usize, DirectTreeActionLog<M>)> = Vec::with_capacity(chunk_size);

    for (index, key) in keys.iter().enumerate() {
        
        let action_log = action_log_store.get(*key).await.unwrap().unwrap();
        pending_chunk.push((index, action_log));
        if pending_chunk.len() >= chunk_size {
            fold_processed_chunk(
                run_parallel_action_log_chunk(
                    &worker_pool,
                    std::mem::take(&mut pending_chunk),
                    advantage_calculation_policy,
                ),
                num_keys,
                &mut finished_samples,
                &mut total_trajectories,
                &mut all_average_absolute_advantages,
                &mut min_heap,
                max_num_training_trajectories,
            );
        }
    }

    if !pending_chunk.is_empty() {
        fold_processed_chunk(
            run_parallel_action_log_chunk(
                &worker_pool,
                pending_chunk,
                advantage_calculation_policy,
            ),
            num_keys,
            &mut finished_samples,
            &mut total_trajectories,
            &mut all_average_absolute_advantages,
            &mut min_heap,
            max_num_training_trajectories,
        );
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
    log_key_value_pair("max_average_absolute_advantage", statistics.max_average_absolute_advantage.to_string());
    log_key_value_pair("min_average_absolute_advantage", statistics.min_average_absolute_advantage.to_string());
    log_key_value_pair("average_absolute_advantage_cutoff", statistics.average_absolute_advantage_cutoff.to_string());
    log_info(format!("total_trajectories: {}, adopted_trajectories: {}", statistics.total_trajectories, statistics.adopted_trajectories));
    kept_trajectories
}

fn run_parallel_action_log_chunk<M: LlmModelMarker>(
    worker_pool: &ThreadPool,
    chunk: Vec<(usize, DirectTreeActionLog<M>)>,
    advantage_calculation_policy: AdvantageCalculationPolicy,
) -> Vec<(usize, Vec<DirectTrainingTrajectory<M>>)> {
    let mut processed_chunk = worker_pool.install(|| {
        chunk
            .into_par_iter()
            .map(|(index, action_log)| {
                (
                    index,
                    action_log_to_candidate_trajectories::<M>(
                        action_log,
                        advantage_calculation_policy,
                    ),
                )
            })
            .collect::<Vec<(usize, Vec<DirectTrainingTrajectory<M>>)>>()
    });
    processed_chunk.sort_by_key(|(index, _)| *index);
    processed_chunk
}

fn fold_processed_chunk<M: LlmModelMarker>(
    processed_chunk: Vec<(usize, Vec<DirectTrainingTrajectory<M>>)>,
    total_samples: usize,
    finished_samples: &mut usize,
    total_trajectories: &mut usize,
    all_average_absolute_advantages: &mut Vec<f32>,
    min_heap: &mut BinaryHeap<Reverse<TrajectoryHeapItem<M>>>,
    max_num_training_trajectories: usize,
) {
    for (_, trajectories) in processed_chunk {
        *finished_samples += 1;
        let progress = if total_samples == 0 {
            0.0
        } else {
            *finished_samples as f32 / total_samples as f32
        };
        log_master_progress(progress, "Rollout Samples Processed");
        fold_candidate_trajectories(
            trajectories,
            total_trajectories,
            all_average_absolute_advantages,
            min_heap,
            max_num_training_trajectories,
        );
    }
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
    advantage_calculation_policy: AdvantageCalculationPolicy,
) -> Vec<DirectTrainingTrajectory<M>> {
    let tree = DirectTree::<M>::from_action_log(&action_log);
    if !tree.completed {
        return Vec::new();
    }
    let root_segment_id = tree
        .root_segment_id
        .expect("DirectTree must have root_segment_id");
    // let mut segment_advantages = tree.calculate_segment_advantages(None);
    let mut segment_advantages = match advantage_calculation_policy {
        AdvantageCalculationPolicy::TreeMappoPosterior => {
            tree.calculate_segment_advantages_from_posteriors(None)
        }
        AdvantageCalculationPolicy::TreeRpoWinRate => {
            tree.calculate_segment_advantages_from_win_rate()
        }
    };
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
    pub advantage_calculation_policy: AdvantageCalculationPolicy,
    pub epoch: usize, // the epoch index
    pub max_num_training_trajectories: usize,
    pub _phantom: std::marker::PhantomData<M>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetFileTrainingTrajectoriesTracking {
    pub rollout_log_hash: Base64Hash,
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub advantage_calculation_policy: AdvantageCalculationPolicy,
    pub epoch: usize, // the epoch index
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
        assert!(
            self.rollout_config.split == DatasetSplit::Training,
            "AssetFileTrainingTrajectories can only be generated from the training split of the rollout logs"
        );
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
            advantage_calculation_policy: self.advantage_calculation_policy.clone(),
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
            log_warning("Training trajectories file is stale or does not exist. Regenerating...");
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
                self.advantage_calculation_policy,
            )
            .await;
            for (i, trajectory) in training_trajectories.into_iter().enumerate() {
                db.upsert(i, &trajectory, SqliteBusyRetryConfig::none())
                    .await
                    .unwrap();
            }
            log_info("Finished generating training trajectories file.");
        }
        write_json(self.version_tracking_path(), &new_tracking_content).unwrap();
        hash_file(self.file_path()).expect("Failed to hash training trajectories file")
    }
}
