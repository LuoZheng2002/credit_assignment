use research_utility::progress_tui_logger::{log_info, log_master_progress};
use std::{collections::BTreeMap, sync::Arc};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    direct_tool::{
        hybrid_dataset::{DatasetSplit, HybridDatasetQuestion, open_hybrid_dataset},
        posterior_calculation_config::PosteriorCalculationConfig,
        rollout_config::DirectRolloutConfig,
        tree::DirectTree,
        tree_action_log::{DirectTreeActionLog, open_action_logs},
    },
    llm_model::LlmModelMarker,
};

use serde::{Deserialize, Serialize};

const DEEPMATH_DATASET_NAME: &str = "deepmath";
const MATH_DATASET_NAME: &str = "math";
const NUMINAMATH_DATASET_NAME: &str = "numinamath";

#[derive(Debug, Clone, Copy)]
struct DatasetBucketStats {
    weighted_num_wins: f32,
    weighted_total_plays: f32,
}

impl DatasetBucketStats {
    fn new() -> Self {
        Self {
            weighted_num_wins: 0.0,
            weighted_total_plays: 0.0,
        }
    }

    fn update(&mut self, num_correct_trajectories: usize, total_trajectories: usize) {
        self.weighted_num_wins += num_correct_trajectories as f32 / total_trajectories as f32;
        self.weighted_total_plays += 1.0;
    }
}

#[derive(Debug, Clone)]
pub struct AccuracyStats {
    pub weighted_num_wins: f32,
    pub weighted_total_plays: f32,
    pub num_trees_with_judgments: usize,
    pub num_trajectories_judged: usize,
    pub deepmath_weighted_num_wins: f32,
    pub deepmath_weighted_total_plays: f32,
    pub math_weighted_num_wins: f32,
    pub math_weighted_total_plays: f32,
    pub numinamath_weighted_num_wins: f32,
    pub numinamath_weighted_total_plays: f32,
}

impl AccuracyStats {
    pub fn accuracy(&self) -> Option<f32> {
        if self.weighted_total_plays == 0.0 {
            None
        } else {
            Some(self.weighted_num_wins / self.weighted_total_plays)
        }
    }

    pub fn accuracy_tuple(&self) -> Option<(f32, f32, f32, f32)> {
        if self.deepmath_weighted_total_plays == 0.0
            || self.math_weighted_total_plays == 0.0
            || self.numinamath_weighted_total_plays == 0.0
        {
            return None;
        }
        let average_accuracy = self.accuracy()?;
        Some((
            average_accuracy,
            self.deepmath_weighted_num_wins / self.deepmath_weighted_total_plays,
            self.math_weighted_num_wins / self.math_weighted_total_plays,
            self.numinamath_weighted_num_wins / self.numinamath_weighted_total_plays,
        ))
    }
}

fn dataset_bucket_name(dataset_name: &str) -> &'static str {
    match dataset_name {
        DEEPMATH_DATASET_NAME => DEEPMATH_DATASET_NAME,
        MATH_DATASET_NAME => MATH_DATASET_NAME,
        NUMINAMATH_DATASET_NAME => NUMINAMATH_DATASET_NAME,
        _ => panic!(
            "Unsupported dataset_name '{}' in get_accuracy; expected one of: deepmath, math, numinamath",
            dataset_name
        ),
    }
}

fn tree_accuracy<M: LlmModelMarker, S: DatasetSplit>(
    action_log: &DirectTreeActionLog<M, S>,
) -> Option<(usize, usize)> {
    let tree = DirectTree::<M, S>::from_action_log(action_log);
    let total_trajectories = tree.leaf_segment_judgments.len();
    if total_trajectories == 0 {
        return None;
    }
    let num_correct_trajectories = tree
        .leaf_segment_judgments
        .values()
        .filter(|judgment| judgment.is_correct)
        .count();
    Some((num_correct_trajectories, total_trajectories))
}

pub async fn get_accuracy<M: LlmModelMarker, S: DatasetSplit>(
    config_nickname: String,
    rollout_config: DirectRolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize,
    progress_bar_label: &str,
) -> AccuracyStats {
    let question_store = open_hybrid_dataset::<S>();
    let question_map: BTreeMap<usize, HybridDatasetQuestion<S>> = question_store
        .iter()
        .expect("failed to iterate hybrid dataset")
        .map(|r| r.expect("failed to read question from hybrid dataset"))
        .collect();
    log_info(format!(
        "get_accuracy: opening action logs for config={config_nickname}, epoch={epoch}"
    ));
    let action_store = open_action_logs::<M, S>(&config_nickname, epoch);
    action_store.sort().unwrap();
    log_info(format!(
        "get_accuracy: action logs opened for config={config_nickname}, epoch={epoch}"
    ));
    let mut keys = action_store.get_keys().unwrap();
    keys.sort();

    log_master_progress(0.0, format!("{}: Calculating", progress_bar_label));

    let num_keys = keys.len();
    let mut weighted_num_wins = 0.0f32;
    let mut weighted_total_plays = 0.0f32;
    let mut num_trees_with_judgments = 0usize;
    let mut num_trajectories_judged = 0usize;
    let mut deepmath_stats = DatasetBucketStats::new();
    let mut math_stats = DatasetBucketStats::new();
    let mut numinamath_stats = DatasetBucketStats::new();

    const MAX_CONCURRENT_TASKS: usize = 200;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_TASKS));
    let mut join_set = JoinSet::new();
    let mut next_key_index = 0usize;

    let mut finished = 0usize;
    while next_key_index < keys.len() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_key_index < keys.len() => {
                let permit = permit_result.expect("accuracy semaphore should not be closed");
                let key = keys[next_key_index];
                next_key_index += 1;

                let question = question_map.get(&key.0).unwrap().clone();
                let dataset_name = dataset_bucket_name(&question.dataset_name).to_string();
                let actions = action_store.load_action_log(key).unwrap();
                let action_log = DirectTreeActionLog {
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    actions,
                };

                join_set.spawn(async move {
                    let _permit = permit;
                    (dataset_name, tree_accuracy::<M, S>(&action_log))
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                finished += 1;

                match joined.expect("join_set must have at least one task") {
                    Ok((dataset_name, result)) => {
                        if let Some((num_correct_trajectories, total_trajectories)) = result {
                            weighted_num_wins +=
                                num_correct_trajectories as f32 / total_trajectories as f32;
                            weighted_total_plays += 1.0;
                            num_trees_with_judgments += 1;
                            num_trajectories_judged += total_trajectories;
                            match dataset_name.as_str() {
                                DEEPMATH_DATASET_NAME => {
                                    deepmath_stats
                                        .update(num_correct_trajectories, total_trajectories)
                                }
                                MATH_DATASET_NAME => {
                                    math_stats.update(num_correct_trajectories, total_trajectories)
                                }
                                NUMINAMATH_DATASET_NAME => {
                                    numinamath_stats.update(num_correct_trajectories, total_trajectories)
                                }
                                _ => unreachable!(
                                    "dataset name was validated before task spawn"
                                ),
                            }
                        }
                    }
                    Err(join_err) => panic!("accuracy task panicked: {join_err}"),
                }

                let progress = if num_keys == 0 {
                    1.0
                } else {
                    finished as f32 / num_keys as f32
                };
                log_master_progress(progress, format!("{}: Calculating", progress_bar_label));
            }
        }
    }

    log_master_progress(1.0, format!("{}: Done", progress_bar_label));

    AccuracyStats {
        weighted_num_wins,
        weighted_total_plays,
        num_trees_with_judgments,
        num_trajectories_judged,
        deepmath_weighted_num_wins: deepmath_stats.weighted_num_wins,
        deepmath_weighted_total_plays: deepmath_stats.weighted_total_plays,
        math_weighted_num_wins: math_stats.weighted_num_wins,
        math_weighted_total_plays: math_stats.weighted_total_plays,
        numinamath_weighted_num_wins: numinamath_stats.weighted_num_wins,
        numinamath_weighted_total_plays: numinamath_stats.weighted_total_plays,
    }
}

pub async fn get_per_question_accuracies<M: LlmModelMarker, S: DatasetSplit>(
    config_nickname: String,
    rollout_config: DirectRolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize,
    progress_bar_label: &str,
) -> Vec<Option<f32>> {
    let question_store = open_hybrid_dataset::<S>();
    let question_map: BTreeMap<usize, HybridDatasetQuestion<S>> = question_store
        .iter()
        .expect("failed to iterate hybrid dataset")
        .map(|r| r.expect("failed to read question from hybrid dataset"))
        .collect();
    let action_store = open_action_logs::<M, S>(&config_nickname, epoch);
    action_store.sort().unwrap();
    let mut keys = action_store.get_keys().unwrap();
    keys.sort();

    log_master_progress(
        0.0,
        format!("{}: Calculating per-question", progress_bar_label),
    );

    let num_keys = keys.len();
    let mut results = vec![None; num_keys];
    let mut finished = 0usize;

    const MAX_CONCURRENT_TASKS: usize = 200;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_TASKS));
    let mut join_set = JoinSet::new();
    let mut next_key_index = 0usize;

    while next_key_index < keys.len() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_key_index < keys.len() => {
                let permit = permit_result.expect("accuracy semaphore should not be closed");
                let key = keys[next_key_index];
                let idx = next_key_index;
                next_key_index += 1;

                let question = question_map.get(&key.0).unwrap().clone();
                let actions = action_store.load_action_log(key).unwrap();
                let action_log = DirectTreeActionLog {
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    actions,
                };

                join_set.spawn(async move {
                    let _permit = permit;
                    let result = tree_accuracy::<M, S>(&action_log);
                    (idx, result)
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                finished += 1;

                match joined.expect("join_set must have at least one task") {
                    Ok((idx, result)) => {
                        if let Some((num_correct, total)) = result {
                            results[idx] = Some(num_correct as f32 / total as f32);
                        }
                    }
                    Err(join_err) => panic!("accuracy task panicked: {join_err}"),
                }

                let progress = if num_keys == 0 {
                    1.0
                } else {
                    finished as f32 / num_keys as f32
                };
                log_master_progress(progress, format!("{}: Calculating per-question", progress_bar_label));
            }
        }
    }

    log_master_progress(1.0, format!("{}: Done", progress_bar_label));
    results
}

fn tree_per_trunk_correctness<M: LlmModelMarker, S: DatasetSplit>(
    action_log: &DirectTreeActionLog<M, S>,
) -> Option<Vec<bool>> {
    let tree = DirectTree::<M, S>::from_action_log(action_log);
    if tree.leaf_segment_judgments.is_empty() {
        return None;
    }
    let correctness: Vec<bool> = tree
        .leaf_segment_judgments
        .values()
        .map(|judgment| judgment.is_correct)
        .collect();
    Some(correctness)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAccuracies {
    pub accuracy_values: Vec<f32>,
    pub mean_accuracy: f32,
    pub confidence_interval_half_width: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAccuracyResult {
    pub per_dataset: BTreeMap<String, DatasetAccuracies>,
}

pub async fn get_test_accuracies<M: LlmModelMarker, S: DatasetSplit>(
    config_nickname: String,
    rollout_config: DirectRolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize,
    progress_bar_label: &str,
    max_num_trunks: usize,
) -> TestAccuracyResult {
    let question_store = open_hybrid_dataset::<S>();
    let question_map: BTreeMap<usize, HybridDatasetQuestion<S>> = question_store
        .iter()
        .expect("failed to iterate hybrid dataset")
        .map(|r| r.expect("failed to read question from hybrid dataset"))
        .collect();
    let action_store = open_action_logs::<M, S>(&config_nickname, epoch);
    action_store.sort().unwrap();
    let mut keys = action_store.get_keys().unwrap();
    keys.sort();

    log_master_progress(
        0.0,
        format!("{}: Calculating per-dataset", progress_bar_label),
    );

    let num_keys = keys.len();
    let mut dataset_per_trunk_correct: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut dataset_total_trees: BTreeMap<String, usize> = BTreeMap::new();
    let mut finished = 0usize;

    const MAX_CONCURRENT_TASKS: usize = 200;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_TASKS));
    let mut join_set = JoinSet::new();
    let mut next_key_index = 0usize;

    while next_key_index < keys.len() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_key_index < keys.len() => {
                let permit = permit_result.expect("accuracy semaphore should not be closed");
                let key = keys[next_key_index];
                next_key_index += 1;

                let question = question_map.get(&key.0).unwrap().clone();
                let dataset_name = question.dataset_name.clone();
                let actions = action_store.load_action_log(key).unwrap();
                let action_log = DirectTreeActionLog {
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    actions,
                };

                join_set.spawn(async move {
                    let _permit = permit;
                    let result = tree_per_trunk_correctness::<M, S>(&action_log);
                    (dataset_name, result)
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                finished += 1;

                match joined.expect("join_set must have at least one task") {
                    Ok((dataset_name, result)) => {
                        if let Some(trunk_correctness) = result {
                            assert_eq!(
                                trunk_correctness.len(), max_num_trunks,
                                "Expected {} trunks per tree but got {}",
                                max_num_trunks, trunk_correctness.len()
                            );
                            let per_trunk = dataset_per_trunk_correct
                                .entry(dataset_name.clone())
                                .or_insert_with(|| vec![0usize; max_num_trunks]);
                            for (i, &is_correct) in trunk_correctness.iter().enumerate() {
                                if is_correct {
                                    per_trunk[i] += 1;
                                }
                            }
                            *dataset_total_trees
                                .entry(dataset_name)
                                .or_insert(0) += 1;
                        }
                    }
                    Err(join_err) => panic!("accuracy task panicked: {join_err}"),
                }

                let progress = if num_keys == 0 {
                    1.0
                } else {
                    finished as f32 / num_keys as f32
                };
                log_master_progress(progress, format!("{}: Calculating per-dataset", progress_bar_label));
            }
        }
    }

    log_master_progress(1.0, format!("{}: Done", progress_bar_label));

    let mut per_dataset = BTreeMap::new();
    for (dataset_name, per_trunk_correct) in dataset_per_trunk_correct {
        let total_trees = dataset_total_trees[&dataset_name] as f32;
        let accuracy_values: Vec<f32> = per_trunk_correct
            .iter()
            .map(|&correct| correct as f32 / total_trees)
            .collect();
        let n = accuracy_values.len() as f32;
        let mean = accuracy_values.iter().sum::<f32>() / n;
        let variance = accuracy_values
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>()
            / (n - 1.0);
        let std_err = (variance / n).sqrt();
        let confidence_interval_half_width = 1.96 * std_err;
        per_dataset.insert(
            dataset_name,
            DatasetAccuracies {
                accuracy_values,
                mean_accuracy: mean,
                confidence_interval_half_width,
            },
        );
    }

    TestAccuracyResult { per_dataset }
}
