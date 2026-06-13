use std::collections::{BTreeMap, BTreeSet};

use std::sync::Arc;

use ordered_float::NotNan;
use research_utility::sqlite_table_array_store::SqliteTableArrayStore;
use research_utility::{
    progress_tui_logger::{log_info, log_key_value_pair, log_master_progress},
    sqlite_store::{SqliteBusyRetryConfig, SqliteStore},
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::direct_tool::hybrid_dataset::{QuestionFlatId, Training, open_hybrid_dataset};
use crate::direct_tool::tree_action::DirectTreeAction;
use crate::{
    direct_tool::{
        hybrid_dataset::{DatasetSplit, HybridDatasetQuestion},
        posterior_calculation_config::PosteriorCalculationConfig,
        rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
        tree::{DirectTree, SegmentContent, SegmentId, TreeCorrectness},
        tree_action_log::{DirectTreeActionLog, open_action_logs},
    },
    jinja_directories::{
        training_trajectories_path_from_template, training_trajectories_stats_path_from_template,
    },
    json_toml_utils::write_json,
    llm_model::LlmModelMarker,
};

const MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH: usize = 8192;
const ALLOW_INCOMPLETE: bool = false;

#[derive(Debug, Clone)]
struct TrajectoryMetadata<S: DatasetSplit> {
    question_flat_id: QuestionFlatId<S>,
    trajectory_index: usize,
    leaf_segment_id: SegmentId,
    average_absolute_advantage: NotNan<f32>,
    trajectory_token_length: usize,
}

#[derive(Debug, Clone)]
struct TrajectorySummary<S: DatasetSplit> {
    question_flat_id: QuestionFlatId<S>,
    leaf_segment_id: SegmentId,
    average_absolute_advantage: f32,
    trajectory_token_length: usize,
}

struct TrajectorySelectionState<S: DatasetSplit> {
    total_samples: usize,
    finished_samples: usize,
    total_trajectories: usize,
    all_average_absolute_advantages: Vec<f32>,
    candidate_metadata: Vec<TrajectoryMetadata<S>>,
    cumulative_avg_abs_advantage_cutoff: f32,
}

impl<S: DatasetSplit> TrajectorySelectionState<S> {
    fn new(total_samples: usize, cumulative_avg_abs_advantage_cutoff: f32) -> Self {
        assert!(
            cumulative_avg_abs_advantage_cutoff > 0.0 && cumulative_avg_abs_advantage_cutoff <= 1.0,
            "cumulative_avg_abs_advantage_cutoff must be in (0.0, 1.0]"
        );
        Self {
            total_samples,
            finished_samples: 0,
            total_trajectories: 0,
            all_average_absolute_advantages: Vec::new(),
            candidate_metadata: Vec::new(),
            cumulative_avg_abs_advantage_cutoff,
        }
    }

    fn into_output(mut self) -> TrainingTrajectorySelectionOutput<S> {
        self.all_average_absolute_advantages
            .sort_by(|a, b| b.partial_cmp(a).unwrap());
        self.candidate_metadata.sort_by(|a, b| {
            b.average_absolute_advantage
                .cmp(&a.average_absolute_advantage)
        });
        let total_average_absolute_advantage_sum: f32 = self
            .candidate_metadata
            .iter()
            .map(|item| *item.average_absolute_advantage)
            .sum();
        let max_selected_average_absolute_advantage_sum =
            self.cumulative_avg_abs_advantage_cutoff * total_average_absolute_advantage_sum;
        let tolerance = f32::EPSILON * total_average_absolute_advantage_sum.max(1.0);
        let mut selected_average_absolute_advantage_sum = 0.0_f32;
        let mut selected_metadata: Vec<TrajectoryMetadata<S>> = Vec::new();
        for item in self.candidate_metadata {
            let next_sum =
                selected_average_absolute_advantage_sum + *item.average_absolute_advantage;
            if next_sum <= max_selected_average_absolute_advantage_sum + tolerance {
                selected_metadata.push(item);
                selected_average_absolute_advantage_sum = next_sum;
            } else {
                break;
            }
        }
        let average_absolute_advantage_cutoff = selected_metadata
            .iter()
            .map(|item| *item.average_absolute_advantage)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        if total_average_absolute_advantage_sum > 0.0 {
            let adopted_share =
                selected_average_absolute_advantage_sum / total_average_absolute_advantage_sum;
            log_key_value_pair(
                "adopted_cumulative_average_absolute_advantage_share",
                format!("{:.6}", adopted_share),
            );
        }
        TrainingTrajectorySelectionOutput {
            selected_metadata,
            all_average_absolute_advantages: self.all_average_absolute_advantages,
            total_trajectories: self.total_trajectories,
            average_absolute_advantage_cutoff,
        }
    }
}

struct TrainingTrajectorySelectionOutput<S: DatasetSplit> {
    selected_metadata: Vec<TrajectoryMetadata<S>>,
    all_average_absolute_advantages: Vec<f32>,
    total_trajectories: usize,
    average_absolute_advantage_cutoff: f32,
}

fn truncate_trajectory_tokens(
    input_ids: &mut Vec<i32>,
    labels: &mut Vec<i32>,
    advantages: &mut Vec<f32>,
) {
    assert_eq!(
        input_ids.len(),
        labels.len(),
        "input_ids and labels must have the same length before truncation"
    );
    assert_eq!(
        input_ids.len(),
        advantages.len(),
        "input_ids and advantages must have the same length before truncation"
    );
    if input_ids.len() <= MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH {
        return;
    }

    let start = input_ids.len() - MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH;
    input_ids.drain(0..start);
    labels.drain(0..start);
    advantages.drain(0..start);

    assert_eq!(
        input_ids.len(),
        labels.len(),
        "input_ids and labels must have the same length after truncation"
    );
    assert_eq!(
        input_ids.len(),
        advantages.len(),
        "input_ids and advantages must have the same length after truncation"
    );
    assert!(
        input_ids.len() <= MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH,
        "trajectory length must be capped after truncation"
    );
}

pub async fn rollout_logs_to_training_trajectories<M: LlmModelMarker>(
    question_store: SqliteStore<QuestionFlatId<Training>, HybridDatasetQuestion<Training>>,
    // action_log_store: DirectTreeActionLogStore<M>,
    action_store: SqliteTableArrayStore<QuestionFlatId<Training>, DirectTreeAction<M>>,
    rollout_config: DirectRolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    training_trajectory_store: SqliteStore<usize, DirectTrainingTrajectory<M>>,
    cumulative_avg_abs_advantage_cutoff: f32,
    statistics_file_path: String,
    advantage_calculation_policy: AdvantageCalculationPolicy,
) {
    assert!(
        cumulative_avg_abs_advantage_cutoff > 0.0 && cumulative_avg_abs_advantage_cutoff <= 1.0,
        "cumulative_avg_abs_advantage_cutoff must be in (0.0, 1.0]"
    );
    // let action_log_store = ActionStoreAdapter::new(action_store);
    let (selection_output, question_store, action_store) =
        select_training_trajectories_from_rollout_logs::<M, Training>(
            question_store,
            action_store,
            rollout_config.clone(),
            posterior_calculation_config.clone(),
            cumulative_avg_abs_advantage_cutoff,
            advantage_calculation_policy,
        )
        .await;
    let TrainingTrajectorySelectionOutput {
        selected_metadata,
        all_average_absolute_advantages,
        total_trajectories,
        average_absolute_advantage_cutoff,
    } = selection_output;
    let adopted_trajectories = materialize_selected_training_trajectories::<M>(
        question_store,
        action_store,
        training_trajectory_store,
        rollout_config,
        posterior_calculation_config,
        selected_metadata,
        advantage_calculation_policy,
    )
    .await;

    let max_average_absolute_advantage = *all_average_absolute_advantages.first().unwrap_or(&0.0);
    let min_average_absolute_advantage = *all_average_absolute_advantages.last().unwrap_or(&0.0);
    // we want to output a histogram and the advantage cutoff
    // and total samples and adopted samples

    let median_average_absolute_advantage = match all_average_absolute_advantages.len() {
        0 => 0.0,
        len if len % 2 == 1 => all_average_absolute_advantages[len / 2],
        len => {
            let upper_mid = len / 2;
            (all_average_absolute_advantages[upper_mid - 1]
                + all_average_absolute_advantages[upper_mid])
                / 2.0
        }
    };

    let statistics = DirectTrainingSetStatistics {
        average_absolute_advantages_sorted: all_average_absolute_advantages,
        max_average_absolute_advantage,
        min_average_absolute_advantage,
        average_absolute_advantage_cutoff,
        total_trajectories,
        adopted_trajectories,
    };
    write_json(statistics_file_path.clone(), &statistics).unwrap();
    log_key_value_pair(
        "max_average_absolute_advantage",
        statistics.max_average_absolute_advantage.to_string(),
    );
    log_key_value_pair(
        "min_average_absolute_advantage",
        statistics.min_average_absolute_advantage.to_string(),
    );
    log_key_value_pair(
        "average_absolute_advantage_cutoff",
        statistics.average_absolute_advantage_cutoff.to_string(),
    );
    log_info(format!(
        "training_samples_generated={} max_average_absolute_advantage={} min_average_absolute_advantage={} median_average_absolute_advantage={}",
        statistics.adopted_trajectories,
        statistics.max_average_absolute_advantage,
        statistics.min_average_absolute_advantage,
        median_average_absolute_advantage,
    ));
}

async fn select_training_trajectories_from_rollout_logs<M: LlmModelMarker, S: DatasetSplit>(
    // action_log_store: &ActionStoreAdapter<M>,
    // action_log_store: DirectTreeActionLogStore<M>,
    question_store: SqliteStore<QuestionFlatId<S>, HybridDatasetQuestion<S>>,
    action_store: SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>>,
    rollout_config: DirectRolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    cumulative_avg_abs_advantage_cutoff: f32,
    advantage_calculation_policy: AdvantageCalculationPolicy,
) -> (
    TrainingTrajectorySelectionOutput<S>,
    SqliteStore<QuestionFlatId<S>, HybridDatasetQuestion<S>>,
    SqliteTableArrayStore<QuestionFlatId<S>, DirectTreeAction<M>>,
) {
    // let mut keys = action_log_store.metadata_store.get_keys().unwrap();
    let mut keys = action_store.get_keys().unwrap();
    let num_keys = keys.len();
    keys.sort();

    let mut selection_state =
        TrajectorySelectionState::new(num_keys, cumulative_avg_abs_advantage_cutoff);
    log_info("Converting action logs to trajectories".to_string());
    let semaphore = Arc::new(Semaphore::new(200));
    let mut join_set = JoinSet::new();
    let mut next_sample_index = 0usize;

    while next_sample_index < keys.len() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_sample_index < keys.len() => {
                let permit = permit_result.expect("selection semaphore should not be closed");
                let sample_index = next_sample_index;
                let key = keys[sample_index];
                next_sample_index += 1;
                // let action_log = action_log_store.get(key).unwrap().unwrap();
                let question = question_store.get(key).unwrap().unwrap();
                let actions = action_store.load_table_sorted(key).unwrap();
                let action_log = DirectTreeActionLog {
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    actions,
                };
                let advantage_calculation_policy = advantage_calculation_policy.clone();
                join_set.spawn(async move {
                    let _permit = permit;
                    let trajectory_summaries = action_log_to_candidate_summaries::<M, S>(
                        action_log,
                        advantage_calculation_policy,
                    );
                    trajectory_summaries
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                match joined.expect("join_set must have at least one task") {
                    Ok(trajectory_summaries) => {
                        selection_state.finished_samples += 1;
                        let progress = if selection_state.total_samples == 0 {
                            0.0
                        } else {
                            selection_state.finished_samples as f32 / selection_state.total_samples as f32
                        };
                        log_master_progress(0.5 * progress, "Phase 1/2: Rollout Samples Processed");

                        for (trajectory_index, trajectory_summary) in trajectory_summaries.into_iter().enumerate() {
                            selection_state.total_trajectories += 1;
                            let average_absolute_advantage =
                                NotNan::new(trajectory_summary.average_absolute_advantage)
                                    .expect("Average absolute segment advantage must not be NaN");
                            selection_state
                                .all_average_absolute_advantages
                                .push(*average_absolute_advantage);

                            selection_state.candidate_metadata.push(TrajectoryMetadata {
                                question_flat_id: trajectory_summary.question_flat_id,
                                trajectory_index,
                                leaf_segment_id: trajectory_summary.leaf_segment_id,
                                average_absolute_advantage,
                                trajectory_token_length: trajectory_summary.trajectory_token_length,
                            });
                        }
                    }
                    Err(join_err) => panic!("selection task panicked: {join_err}"),
                }
            }
        }
    }

    (selection_state.into_output(), question_store, action_store)
}

async fn materialize_selected_training_trajectories<M: LlmModelMarker>(
    question_store: SqliteStore<QuestionFlatId<Training>, HybridDatasetQuestion<Training>>,
    action_store: SqliteTableArrayStore<QuestionFlatId<Training>, DirectTreeAction<M>>,
    training_trajectory_store: SqliteStore<usize, DirectTrainingTrajectory<M>>,
    rollout_config: DirectRolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    mut selected_metadata: Vec<TrajectoryMetadata<Training>>,
    advantage_calculation_policy: AdvantageCalculationPolicy,
) -> usize {
    selected_metadata.sort_by(|a, b| {
        b.trajectory_token_length
            .cmp(&a.trajectory_token_length)
            .then_with(|| {
                b.average_absolute_advantage
                    .cmp(&a.average_absolute_advantage)
            })
            .then_with(|| a.question_flat_id.cmp(&b.question_flat_id))
            .then_with(|| a.trajectory_index.cmp(&b.trajectory_index))
    });

    let adopted_trajectories = selected_metadata.len();
    assert!(
        adopted_trajectories > 0,
        "cumulative_avg_abs_advantage_cutoff kept zero trajectories; increase cutoff"
    );

    let mut grouped_metadata: BTreeMap<
        QuestionFlatId<Training>,
        Vec<(usize, TrajectoryMetadata<Training>)>,
    > = BTreeMap::new();
    for (output_index, metadata) in selected_metadata.into_iter().enumerate() {
        grouped_metadata
            .entry(metadata.question_flat_id)
            .or_default()
            .push((output_index, metadata));
    }

    let semaphore = Arc::new(Semaphore::new(200));
    let mut join_set = JoinSet::new();
    let mut finished_trajectories = 0usize;
    let tolerance = 10.0 * f32::EPSILON;
    while !grouped_metadata.is_empty() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if !grouped_metadata.is_empty() => {
                let permit = permit_result.expect("materialization semaphore should not be closed");
                let (question_flat_id, selected_for_question) = grouped_metadata
                    .pop_first()
                    .expect("grouped metadata should be non-empty");
                let key = question_flat_id;
                // let action_log = action_log_store.get(*key).unwrap().unwrap();
                let question = question_store.get(key).unwrap().unwrap();
                let actions = action_store.load_table_sorted(key).unwrap();
                let action_log = DirectTreeActionLog{
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    actions,
                };
                let advantage_calculation_policy = advantage_calculation_policy.clone();
                join_set.spawn(async move {
                    let _permit = permit;
                    let selected_trajectory_indices: BTreeSet<usize> = selected_for_question
                        .iter()
                        .map(|(_, metadata)| metadata.trajectory_index)
                        .collect();
                    assert_eq!(
                        selected_trajectory_indices.len(),
                        selected_for_question.len(),
                        "duplicate trajectory_index selected for question_flat_id {:?}",
                        question_flat_id,
                    );
                    let reconstructed_trajectories = action_log_to_selected_trajectories::<M>(
                        action_log,
                        advantage_calculation_policy,
                        &selected_trajectory_indices,
                    );
                    assert_eq!(
                        reconstructed_trajectories.len(),
                        selected_for_question.len(),
                        "failed to reconstruct selected trajectories for question_flat_id {:?}; selected={}, reconstructed={}",
                        question_flat_id,
                        selected_for_question.len(),
                        reconstructed_trajectories.len(),
                    );

                    let mut reconstructed_by_trajectory_index: BTreeMap<usize, DirectTrainingTrajectory<M>> =
                        reconstructed_trajectories.into_iter().collect();
                    let mut outputs: Vec<(usize, DirectTrainingTrajectory<M>)> =
                        Vec::with_capacity(selected_for_question.len());

                    for (output_index, metadata) in selected_for_question.into_iter() {
                        let trajectory = reconstructed_by_trajectory_index
                            .remove(&metadata.trajectory_index)
                            .expect("selected trajectory index should have been reconstructed");
                        assert_eq!(
                            trajectory.question.flat_id, metadata.question_flat_id,
                            "reconstructed question flat_id mismatch at question_flat_id {:?}, trajectory index {}",
                            metadata.question_flat_id, metadata.trajectory_index
                        );
                        assert_eq!(
                            trajectory.leaf_segment_id, metadata.leaf_segment_id,
                            "reconstructed leaf segment id mismatch at question_flat_id {:?}, trajectory index {}",
                            metadata.question_flat_id, metadata.trajectory_index
                        );
                        let expected_average_absolute_advantage = *metadata.average_absolute_advantage;
                        let diff = (trajectory.average_absolute_segment_advantage
                            - expected_average_absolute_advantage)
                            .abs();
                        assert!(
                            diff <= tolerance,
                            "reconstructed average absolute advantage mismatch at question_flat_id {:?}, trajectory index {}: expected {}, got {}, diff {}",
                            metadata.question_flat_id,
                            metadata.trajectory_index,
                            expected_average_absolute_advantage,
                            trajectory.average_absolute_segment_advantage,
                            diff
                        );
                        assert_eq!(
                            trajectory.input_ids.len(),
                            metadata.trajectory_token_length,
                            "reconstructed token length mismatch at question_flat_id {:?}, trajectory index {}",
                            metadata.question_flat_id,
                            metadata.trajectory_index
                        );
                        assert_eq!(
                            trajectory.input_ids.len(),
                            trajectory.labels.len(),
                            "kept trajectory must satisfy input_ids.len() == labels.len(); question_flat_id={:?}",
                            trajectory.question.flat_id
                        );
                        assert_eq!(
                            trajectory.input_ids.len(),
                            trajectory.advantages.len(),
                            "kept trajectory must satisfy input_ids.len() == advantages.len(); question_flat_id={:?}",
                            trajectory.question.flat_id
                        );
                        outputs.push((output_index, trajectory));
                    }
                    assert!(
                        reconstructed_by_trajectory_index.is_empty(),
                        "unexpected reconstructed trajectories left over for question_flat_id {:?}",
                        question_flat_id,
                    );
                    outputs
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                match joined.expect("join_set must have at least one task") {
                    Ok(trajectories) => {
                        for (output_index, trajectory) in trajectories {
                            training_trajectory_store
                                .upsert(output_index, &trajectory, SqliteBusyRetryConfig::none())
                                .unwrap();
                            finished_trajectories += 1;
                            let progress = 0.5
                                + 0.5 * (finished_trajectories as f32 / adopted_trajectories as f32);
                            log_master_progress(progress, "Phase 2/2: Selected Trajectories Materialized");
                        }
                    }
                    Err(join_err) => panic!("materialization task panicked: {join_err}"),
                }
            }
        }
    }
    log_master_progress(1.0, "Phase 2/2: Selected Trajectories Materialized");

    adopted_trajectories
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

fn action_log_to_candidate_summaries<M: LlmModelMarker, S: DatasetSplit>(
    action_log: DirectTreeActionLog<M, S>,
    advantage_calculation_policy: AdvantageCalculationPolicy,
) -> Vec<TrajectorySummary<S>> {
    let tree = DirectTree::from_action_log(&action_log);
    if !ALLOW_INCOMPLETE && !tree.completed() {
        return Vec::new();
    }
    if !matches!(tree.get_correctness(), TreeCorrectness::Mixed) {
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
    let mut trajectory_summaries: Vec<TrajectorySummary<S>> = Vec::new();
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
        let mut sum_absolute_advantage = 0.0;
        let mut non_root_segment_count = 0usize;
        let mut trajectory_token_length = 0usize;
        for segment_id in segment_ids.iter() {
            let segment = tree.segments.get(segment_id).unwrap();
            let segment_advantage = segment_advantages.get_mut(segment_id).unwrap();
            if *segment_id != root_segment_id {
                sum_absolute_advantage += segment_advantage.abs();
                non_root_segment_count += 1;
            }
            for content in segment.content.iter() {
                let token_count = match content {
                    SegmentContent::Prompt(token_array)
                    | SegmentContent::ToolResponse(token_array) => token_array.tokens.len(),
                    SegmentContent::ReasoningOrToolCall {
                        tokens,
                        complete: _,
                    } => tokens.tokens.len(),
                };
                trajectory_token_length += token_count;
            }
            *segment_advantage = 0.0; // we set the advantage of the taken segments to 0
        }
        let average_absolute_advantage =
            sum_absolute_advantage / non_root_segment_count.max(1) as f32;
        assert_eq!(average_absolute_advantage, best_average_absolute_advantage);
        trajectory_summaries.push(TrajectorySummary {
            question_flat_id: tree.action_log.question.flat_id,
            leaf_segment_id: best_leaf,
            average_absolute_advantage,
            trajectory_token_length: trajectory_token_length
                .min(MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH),
        });
        leaf_segment_ids.remove(&best_leaf);
    }
    trajectory_summaries
}

fn action_log_to_selected_trajectories<M: LlmModelMarker>(
    action_log: DirectTreeActionLog<M, Training>,
    advantage_calculation_policy: AdvantageCalculationPolicy,
    selected_trajectory_indices: &BTreeSet<usize>,
) -> Vec<(usize, DirectTrainingTrajectory<M>)> {
    if selected_trajectory_indices.is_empty() {
        return Vec::new();
    }
    let tree = DirectTree::<M, Training>::from_action_log(&action_log);
    if !ALLOW_INCOMPLETE && !tree.completed() {
        return Vec::new();
    }
    if !matches!(tree.get_correctness(), TreeCorrectness::Mixed) {
        return Vec::new();
    }
    let root_segment_id = tree
        .root_segment_id
        .expect("DirectTree must have root_segment_id");
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
    let max_selected_trajectory_index = *selected_trajectory_indices
        .iter()
        .max()
        .expect("selected trajectory indices should be non-empty");
    let mut reconstructed: Vec<(usize, DirectTrainingTrajectory<M>)> = Vec::new();
    let mut leaf_segment_ids: BTreeSet<SegmentId> =
        tree.leaf_segment_judgments.keys().cloned().collect();
    let mut trajectory_index = 0usize;
    while !leaf_segment_ids.is_empty() && trajectory_index <= max_selected_trajectory_index {
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

        let should_materialize = selected_trajectory_indices.contains(&trajectory_index);
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
            if should_materialize {
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
            }
            *segment_advantage = 0.0; // we set the advantage of the taken segments to 0
        }
        let average_absolute_advantage =
            sum_absolute_advantage / non_root_segment_count.max(1) as f32;
        assert_eq!(average_absolute_advantage, best_average_absolute_advantage);

        if should_materialize {
            truncate_trajectory_tokens(&mut input_ids, &mut labels, &mut advantages);
            assert!(
                input_ids.len() >= 2,
                "trajectory must contain at least two tokens"
            );
            assert!(
                labels.iter().skip(1).any(|label| *label != -100),
                "trajectory must contain at least one supervised token after causal shift"
            );
            reconstructed.push((
                trajectory_index,
                DirectTrainingTrajectory {
                    question: tree.action_log.question.clone(),
                    leaf_segment_id: best_leaf,
                    input_ids,
                    labels,
                    advantages,
                    average_absolute_segment_advantage: average_absolute_advantage,
                    _phantom: std::marker::PhantomData::<M>,
                },
            ));
        }
        leaf_segment_ids.remove(&best_leaf);
        trajectory_index += 1;
    }

    reconstructed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectTrainingTrajectory<M: LlmModelMarker> {
    pub question: HybridDatasetQuestion<Training>,
    pub leaf_segment_id: SegmentId,
    pub input_ids: Vec<i32>,
    pub labels: Vec<i32>, // we may not need to let model learn to stop at tool-call boundaries or end since our framework already handled this
    pub advantages: Vec<f32>,
    pub average_absolute_segment_advantage: f32,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<M>,
}

fn training_trajectories_short_hash(
    config_nickname: &str,
    rollout_config: &DirectRolloutConfig<Training>,
    posterior_calculation_config: &PosteriorCalculationConfig,
    epoch: usize,
) -> String {
    let serialized = serde_json::to_vec(&(
        &config_nickname,
        rollout_config,
        posterior_calculation_config,
        &epoch,
    ))
    .unwrap();
    let hash = blake3::hash(&serialized);
    hex::encode(&hash.as_bytes()[..4])
}

pub fn training_trajectories_file_path<M: LlmModelMarker>(
    config_nickname: &str,
    rollout_config: &DirectRolloutConfig<Training>,
    posterior_calculation_config: &PosteriorCalculationConfig,
    epoch: usize,
) -> String {
    let hash = training_trajectories_short_hash(
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
    );
    training_trajectories_path_from_template(M::CLI_NAME, config_nickname, epoch, &hash)
        .unwrap_or_else(|err| {
            panic!(
                "failed to render training trajectories path for model_cli_name={}, config_nickname={}, epoch={}, hash={}: {}",
                M::CLI_NAME, config_nickname, epoch, hash, err
            )
        })
}

pub fn training_trajectories_stats_file_path<M: LlmModelMarker>(
    config_nickname: &str,
    rollout_config: &DirectRolloutConfig<Training>,
    posterior_calculation_config: &PosteriorCalculationConfig,
    epoch: usize,
) -> String {
    let hash = training_trajectories_short_hash(
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
    );
    training_trajectories_stats_path_from_template(M::CLI_NAME, config_nickname, epoch, &hash)
        .unwrap_or_else(|err| {
            panic!(
                "failed to render training trajectories stats path for model_cli_name={}, config_nickname={}, epoch={}, hash={}: {}",
                M::CLI_NAME, config_nickname, epoch, hash, err
            )
        })
}

pub fn open_training_trajectories<M: LlmModelMarker>(
    config_nickname: &str,
    rollout_config: &DirectRolloutConfig<Training>,
    posterior_calculation_config: &PosteriorCalculationConfig,
    epoch: usize,
) -> SqliteStore<usize, DirectTrainingTrajectory<M>> {
    SqliteStore::<usize, DirectTrainingTrajectory<M>>::assume_initialized(
        training_trajectories_file_path::<M>(
            config_nickname,
            rollout_config,
            posterior_calculation_config,
            epoch,
        ),
        false,
    )
    .unwrap_or_else(|e| {
        panic!(
            "Failed to open training trajectories sqlite store at {}: {}",
            training_trajectories_file_path::<M>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                epoch,
            ),
            e
        )
    })
}

pub async fn generate_training_trajectories<M: LlmModelMarker>(
    config_nickname: &str,
    rollout_config: DirectRolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize,
    cumulative_avg_abs_advantage_cutoff: f32,
    advantage_calculation_policy: AdvantageCalculationPolicy,
) -> SqliteStore<usize, DirectTrainingTrajectory<M>> {
    let file_path = training_trajectories_file_path::<M>(
        config_nickname,
        &rollout_config,
        &posterior_calculation_config,
        epoch,
    );
    if std::path::Path::new(&file_path).exists() {
        std::fs::remove_file(&file_path).unwrap();
    }
    let stats_file_path = training_trajectories_stats_file_path::<M>(
        config_nickname,
        &rollout_config,
        &posterior_calculation_config,
        epoch,
    );
    let training_trajectory_store =
        SqliteStore::<usize, DirectTrainingTrajectory<M>>::initialize(file_path.clone())
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to initialize training trajectories sqlite store at {}: {}",
                    file_path, e
                )
            });
    let dataset_store = open_hybrid_dataset::<Training>();
    let action_store = open_action_logs::<M, Training>(config_nickname, epoch);
    rollout_logs_to_training_trajectories::<M>(
        dataset_store,
        action_store,
        rollout_config,
        posterior_calculation_config,
        training_trajectory_store,
        cumulative_avg_abs_advantage_cutoff,
        stats_file_path,
        advantage_calculation_policy,
    )
    .await;
    log_info("Finished generating training trajectories file.");
    SqliteStore::<usize, DirectTrainingTrajectory<M>>::assume_initialized(file_path, false)
        .unwrap_or_else(|e| panic!("Failed to reopen training trajectories sqlite store: {}", e))
}
