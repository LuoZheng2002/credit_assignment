use research_utility::progress_text_logger::{log_info, log_master_progress};
use std::{collections::BTreeMap, sync::Arc};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    constants::temperature_by_split,
    hybrid_dataset::{DatasetSplit, HybridDatasetQuestion, QuestionFlatId, open_hybrid_dataset},
    llm_model::LlmModelMarker,
    posterior_calculation_config::PosteriorCalculationConfig,
    rollout_config::RolloutConfig,
    tree::DirectTree,
    tree_action_log::{ActionLogStore, DirectTreeActionLog, open_action_logs},
    tree_artifact::{TreeJudged, load_available_tree_judged_artifacts, load_tree_judged_artifacts},
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

fn tree_judged_accuracy<M: LlmModelMarker, S: DatasetSplit>(
    tree_judged: &TreeJudged<M, S>,
) -> Option<(usize, usize)> {
    let total_trajectories = tree_judged.tree.leaf_answers.len();
    if total_trajectories == 0 {
        return None;
    }
    let num_correct_trajectories = tree_judged
        .tree
        .leaf_answers
        .iter()
        .filter(|leaf| {
            tree_judged
                .judgment
                .judgments_by_model_answer
                .get(&leaf.model_answer_string)
                .copied()
                .unwrap_or(false)
        })
        .count();
    Some((num_correct_trajectories, total_trajectories))
}

fn build_question_map<S: DatasetSplit>() -> BTreeMap<usize, HybridDatasetQuestion<S>> {
    let question_store = open_hybrid_dataset::<S>();
    question_store
        .iter()
        .expect("failed to iterate hybrid dataset")
        .map(|r| r.expect("failed to read question from hybrid dataset"))
        .collect()
}

pub async fn get_accuracy_from_tree_judgments_at_path<M: LlmModelMarker, S: DatasetSplit>(
    tree_artifacts_path: &str,
    tree_judgment_jsonl_path: &str,
    progress_bar_label: &str,
) -> AccuracyStats {
    let tree_judged_artifacts =
        load_available_tree_judged_artifacts::<M, S>(tree_artifacts_path, tree_judgment_jsonl_path)
            .unwrap_or_else(|err| panic!("failed to load judged tree artifacts: {}", err));
    let num_keys = tree_judged_artifacts.len();
    assert!(
        num_keys > 0,
        "No judged tree artifacts loaded from tree_artifacts_path={} tree_judgment_jsonl_path={}; refusing to report a zero accuracy for missing artifacts",
        tree_artifacts_path,
        tree_judgment_jsonl_path
    );
    let mut weighted_num_wins = 0.0f32;
    let mut weighted_total_plays = 0.0f32;
    let mut num_trees_with_judgments = 0usize;
    let mut num_trajectories_judged = 0usize;
    let mut deepmath_stats = DatasetBucketStats::new();
    let mut math_stats = DatasetBucketStats::new();
    let mut numinamath_stats = DatasetBucketStats::new();

    log_master_progress(0.0, format!("{}: Calculating", progress_bar_label));
    for (index, tree_judged) in tree_judged_artifacts.iter().enumerate() {
        let dataset_name = dataset_bucket_name(&tree_judged.tree.question.dataset_name);
        if let Some((num_correct_trajectories, total_trajectories)) =
            tree_judged_accuracy(tree_judged)
        {
            weighted_num_wins += num_correct_trajectories as f32 / total_trajectories as f32;
            weighted_total_plays += 1.0;
            num_trees_with_judgments += 1;
            num_trajectories_judged += total_trajectories;
            match dataset_name {
                DEEPMATH_DATASET_NAME => {
                    deepmath_stats.update(num_correct_trajectories, total_trajectories)
                }
                MATH_DATASET_NAME => {
                    math_stats.update(num_correct_trajectories, total_trajectories)
                }
                NUMINAMATH_DATASET_NAME => {
                    numinamath_stats.update(num_correct_trajectories, total_trajectories)
                }
                _ => unreachable!("dataset name was validated"),
            }
        }
        let progress = if num_keys == 0 {
            1.0
        } else {
            (index + 1) as f32 / num_keys as f32
        };
        log_master_progress(progress, format!("{}: Calculating", progress_bar_label));
    }
    log_master_progress(1.0, format!("{}: Done", progress_bar_label));
    assert!(
        num_trees_with_judgments > 0,
        "No trees with judgments were scored from tree_artifacts_path={} tree_judgment_jsonl_path={}; refusing to report a zero accuracy for missing judgments",
        tree_artifacts_path,
        tree_judgment_jsonl_path
    );

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

async fn compute_accuracy_stats<M: LlmModelMarker, S: DatasetSplit>(
    keys: Vec<QuestionFlatId<S>>,
    question_map: &BTreeMap<usize, HybridDatasetQuestion<S>>,
    action_store: &ActionLogStore<M, S>,
    rollout_config: RolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    progress_bar_label: &str,
    use_tool: bool,
    config_nickname: String,
) -> AccuracyStats {
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
                    mount_dir: String::new(),
                    config_nickname: config_nickname.clone(),
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    use_tool,
                    fixed_temperature: temperature_by_split::<S>(),
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

pub async fn get_accuracy<M: LlmModelMarker, S: DatasetSplit>(
    mount_dir: &str,
    config_nickname: String,
    rollout_config: RolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize,
    progress_bar_label: &str,
    use_tool: bool,
) -> AccuracyStats {
    let question_map = build_question_map::<S>();
    log_info(format!(
        "get_accuracy: opening action logs for config={config_nickname}, epoch={epoch}"
    ));
    let action_store = open_action_logs::<M, S>(mount_dir, &config_nickname, epoch);
    action_store.sort().unwrap();
    log_info(format!(
        "get_accuracy: action logs opened for config={config_nickname}, epoch={epoch}"
    ));
    let mut keys = action_store.get_keys().unwrap();
    keys.sort();

    log_master_progress(0.0, format!("{}: Calculating", progress_bar_label));

    compute_accuracy_stats(
        keys,
        &question_map,
        &action_store,
        rollout_config,
        posterior_calculation_config,
        progress_bar_label,
        use_tool,
        config_nickname,
    )
    .await
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

fn tree_judged_per_trunk_correctness<M: LlmModelMarker, S: DatasetSplit>(
    tree_judged: &TreeJudged<M, S>,
) -> Option<Vec<bool>> {
    if tree_judged.tree.leaf_answers.is_empty() {
        return None;
    }
    Some(
        tree_judged
            .tree
            .leaf_answers
            .iter()
            .map(|leaf| {
                tree_judged
                    .judgment
                    .judgments_by_model_answer
                    .get(&leaf.model_answer_string)
                    .copied()
                    .unwrap_or(false)
            })
            .collect(),
    )
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
    pub macro_average: DatasetAccuracies,
}

fn summarize_accuracy_values(accuracy_values: Vec<f32>) -> DatasetAccuracies {
    let n = accuracy_values.len() as f32;
    let mean = if n > 0.0 {
        accuracy_values.iter().sum::<f32>() / n
    } else {
        0.0
    };
    let variance = if n > 1.0 {
        accuracy_values
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>()
            / (n - 1.0)
    } else {
        0.0
    };
    let std_err = if n > 0.0 { (variance / n).sqrt() } else { 0.0 };
    DatasetAccuracies {
        accuracy_values,
        mean_accuracy: mean,
        confidence_interval_half_width: 1.96 * std_err,
    }
}

pub fn equal_dataset_macro_average(
    per_dataset: &BTreeMap<String, DatasetAccuracies>,
) -> DatasetAccuracies {
    let max_trials = per_dataset
        .values()
        .map(|accuracies| accuracies.accuracy_values.len())
        .max()
        .unwrap_or(0);
    let mut macro_values = Vec::new();
    for trial_index in 0..max_trials {
        let values = per_dataset
            .values()
            .filter_map(|accuracies| accuracies.accuracy_values.get(trial_index))
            .copied()
            .collect::<Vec<_>>();
        if !values.is_empty() {
            macro_values.push(values.iter().sum::<f32>() / values.len() as f32);
        }
    }
    summarize_accuracy_values(macro_values)
}

pub async fn get_test_accuracies<M: LlmModelMarker, S: DatasetSplit>(
    mount_dir: &str,
    config_nickname: String,
    rollout_config: RolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize,
    progress_bar_label: &str,
    num_trunks: usize,
    use_tool: bool,
) -> TestAccuracyResult {
    let question_map = build_question_map::<S>();
    let action_store = open_action_logs::<M, S>(mount_dir, &config_nickname, epoch);
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
                    mount_dir: String::new(),
                    config_nickname: config_nickname.clone(),
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    use_tool,
                    fixed_temperature: temperature_by_split::<S>(),
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
                                trunk_correctness.len(), num_trunks,
                                "Expected {} trunks per tree but got {}",
                                num_trunks, trunk_correctness.len()
                            );
                            let per_trunk = dataset_per_trunk_correct
                                .entry(dataset_name.clone())
                                .or_insert_with(|| vec![0usize; num_trunks]);
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
        per_dataset.insert(dataset_name, summarize_accuracy_values(accuracy_values));
    }

    let macro_average = equal_dataset_macro_average(&per_dataset);
    TestAccuracyResult {
        per_dataset,
        macro_average,
    }
}

pub async fn get_test_accuracies_from_tree_judgments_at_path<M: LlmModelMarker, S: DatasetSplit>(
    tree_artifacts_path: &str,
    tree_judgment_jsonl_path: &str,
    progress_bar_label: &str,
    num_trunks: usize,
) -> TestAccuracyResult {
    let tree_judged_artifacts =
        load_tree_judged_artifacts::<M, S>(tree_artifacts_path, tree_judgment_jsonl_path)
            .unwrap_or_else(|err| panic!("failed to load judged test tree artifacts: {}", err));

    log_master_progress(
        0.0,
        format!("{}: Calculating per-dataset", progress_bar_label),
    );

    let num_keys = tree_judged_artifacts.len();
    let mut dataset_per_trunk_correct: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut dataset_total_trees: BTreeMap<String, usize> = BTreeMap::new();

    for (index, tree_judged) in tree_judged_artifacts.iter().enumerate() {
        if let Some(trunk_correctness) = tree_judged_per_trunk_correctness(tree_judged) {
            assert_eq!(
                trunk_correctness.len(),
                num_trunks,
                "Expected {} trunks per tree but got {}",
                num_trunks,
                trunk_correctness.len()
            );
            let dataset_name = tree_judged.tree.question.dataset_name.clone();
            let per_trunk = dataset_per_trunk_correct
                .entry(dataset_name.clone())
                .or_insert_with(|| vec![0usize; num_trunks]);
            for (trunk_index, &is_correct) in trunk_correctness.iter().enumerate() {
                if is_correct {
                    per_trunk[trunk_index] += 1;
                }
            }
            *dataset_total_trees.entry(dataset_name).or_insert(0) += 1;
        }
        let progress = if num_keys == 0 {
            1.0
        } else {
            (index + 1) as f32 / num_keys as f32
        };
        log_master_progress(
            progress,
            format!("{}: Calculating per-dataset", progress_bar_label),
        );
    }

    log_master_progress(1.0, format!("{}: Done", progress_bar_label));

    let mut per_dataset = BTreeMap::new();
    for (dataset_name, per_trunk_correct) in dataset_per_trunk_correct {
        let total_trees = dataset_total_trees[&dataset_name] as f32;
        let accuracy_values: Vec<f32> = per_trunk_correct
            .iter()
            .map(|&correct| correct as f32 / total_trees)
            .collect();
        per_dataset.insert(dataset_name, summarize_accuracy_values(accuracy_values));
    }

    let macro_average = equal_dataset_macro_average(&per_dataset);
    TestAccuracyResult {
        per_dataset,
        macro_average,
    }
}

/// Like `get_accuracy` but reads action logs from an explicit store path.
pub async fn get_accuracy_at_path<M: LlmModelMarker, S: DatasetSplit>(
    action_log_store_path: &str,
    rollout_config: RolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    progress_bar_label: &str,
    use_tool: bool,
) -> AccuracyStats {
    let question_map = build_question_map::<S>();
    log_info(format!(
        "get_accuracy_at_path: opening action logs at {action_log_store_path}"
    ));
    let action_store =
        ActionLogStore::<M, S>::initialize_if_missing(action_log_store_path.to_string())
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to open action log store at {}: {}",
                    action_log_store_path, e
                )
            });
    action_store.sort().unwrap();
    let mut keys = action_store.get_keys().unwrap();
    keys.sort();

    log_master_progress(0.0, format!("{}: Calculating", progress_bar_label));

    compute_accuracy_stats(
        keys,
        &question_map,
        &action_store,
        rollout_config,
        posterior_calculation_config,
        progress_bar_label,
        use_tool,
        "".to_string(),
    )
    .await
}
