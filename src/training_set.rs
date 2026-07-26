use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Cursor, Read, Write};
use std::path::Path;
use std::sync::Arc;

use ordered_float::NotNan;
use rand::seq::SliceRandom;
use research_utility::progress_text_logger::{
    log_info, log_key_value_pair, log_master_progress, log_warning,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::hybrid_dataset::{QuestionFlatId, Training, open_hybrid_dataset};

use crate::{
    constants,
    directories::{training_trajectories_path, training_trajectories_stats_path},
    hybrid_dataset::{DatasetSplit, HybridDatasetQuestion},
    json_toml_utils::write_json,
    llm_model::LlmModelMarker,
    posterior_calculation_config::PosteriorCalculationConfig,
    rollout_config::{RolloutConfig, TrainingAdvantagePolicy},
    tree::{DirectTree, SegmentContent, SegmentId, TreeCorrectness},
    tree_action_log::{
        ActionLogConfigBundle, ActionLogStore, DirectTreeActionLog, open_action_logs,
    },
};

const MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH: usize = 8192;
const ALLOW_INCOMPLETE: bool = false;
const MAX_ABSOLUTE_TOKEN_ADVANTAGE: f32 = 3.0;
const EXTREME_ABSOLUTE_TOKEN_ADVANTAGE_WARNING_THRESHOLD: f32 = 5.0;
const ADVANTAGE_BALANCE_EPSILON: f32 = 1.0e-12;
const INV_CDF_ADVANTAGE_THRESHOLD: f32 = 0.05;
const CUTOFF_FRACTION: f32 = 0.1;
const MIN_ADOPTED_TRAJECTORY_FRACTION: f32 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
pub enum TrainingSetSortMode {
    /// Sort by trajectory token length ascending (shortest first), then by
    /// average absolute advantage descending.
    #[serde(alias = "by_length_ascending")]
    ByLengthAscending,
    /// Sort by trajectory token length descending (longest first), then by
    /// average absolute advantage descending. This is the original/default behavior.
    #[serde(
        alias = "ByLength",
        alias = "by_length",
        alias = "by_length_descending"
    )]
    ByLengthDescending,
    /// Follow the order in the action log: grouped by question (ascending
    /// question flat id), then by trajectory index (descending average
    /// absolute advantage within each question).
    ByQuestion,
    /// Shuffle the entire training set using a random number generator.
    RandomShuffle,
}

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
}

impl<S: DatasetSplit> TrajectorySelectionState<S> {
    fn new(total_samples: usize) -> Self {
        Self {
            total_samples,
            finished_samples: 0,
            total_trajectories: 0,
            all_average_absolute_advantages: Vec::new(),
            candidate_metadata: Vec::new(),
        }
    }

    fn into_output(mut self) -> TrainingTrajectorySelectionOutput<S> {
        self.all_average_absolute_advantages
            .sort_by(|a, b| b.partial_cmp(a).unwrap());
        self.candidate_metadata.sort_by(|a, b| {
            b.average_absolute_advantage
                .cmp(&a.average_absolute_advantage)
        });
        let total_trajectories = self.total_trajectories;
        let total_average_absolute_advantage_sum: f32 = self
            .candidate_metadata
            .iter()
            .map(|item| *item.average_absolute_advantage)
            .sum();

        // Find the inverse CDF 5% threshold: the average absolute advantage at
        // the point where trajectories better than this one account for 5% of
        // total advantage.
        let target_advantage = INV_CDF_ADVANTAGE_THRESHOLD * total_average_absolute_advantage_sum;
        let tolerance = f32::EPSILON * total_average_absolute_advantage_sum.max(1.0);
        let mut cumulative_adv = 0.0_f32;
        let mut inv_cdf_5pct_threshold = 0.0_f32;
        for item in self.candidate_metadata.iter() {
            cumulative_adv += *item.average_absolute_advantage;
            if cumulative_adv > target_advantage + tolerance {
                inv_cdf_5pct_threshold = *item.average_absolute_advantage;
                break;
            }
        }
        // If we never crossed the threshold (e.g. only 1 trajectory or all
        // zero), keep all trajectories.
        if inv_cdf_5pct_threshold <= 0.0 {
            inv_cdf_5pct_threshold = self
                .candidate_metadata
                .last()
                .map(|item| *item.average_absolute_advantage)
                .unwrap_or(0.0);
        }

        let cutoff = inv_cdf_5pct_threshold * CUTOFF_FRACTION;

        // Keep trajectories with average absolute advantage >= cutoff.
        let mut selected_metadata: Vec<TrajectoryMetadata<S>> = Vec::new();
        for item in self.candidate_metadata.iter() {
            if *item.average_absolute_advantage >= cutoff {
                selected_metadata.push(item.clone());
            }
        }

        // Patch: ensure at least MIN_ADOPTED_TRAJECTORY_FRACTION of all
        // trajectories are kept, to avoid trimming too much on a long, narrow
        // tail.
        let min_adopted =
            (total_trajectories as f32 * MIN_ADOPTED_TRAJECTORY_FRACTION).ceil() as usize;
        let min_adopted = min_adopted.min(total_trajectories);
        if selected_metadata.len() < min_adopted {
            // The candidate_metadata is already sorted descending, so take the
            // top min_adopted.
            selected_metadata = self.candidate_metadata[..min_adopted].to_vec();
        }

        let adopted_trajectories = selected_metadata.len();
        let adopted_percentage = if total_trajectories > 0 {
            100.0 * adopted_trajectories as f32 / total_trajectories as f32
        } else {
            0.0
        };
        let lowest_average_absolute_advantage_adopted = selected_metadata
            .iter()
            .map(|item| *item.average_absolute_advantage)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let average_absolute_advantage_cutoff = lowest_average_absolute_advantage_adopted;

        log_info(format!(
            "adopted_trajectories={} total_trajectories={} adopted_percentage={:.2}% lowest_average_absolute_advantage_adopted={:.6}",
            adopted_trajectories,
            total_trajectories,
            adopted_percentage,
            lowest_average_absolute_advantage_adopted,
        ));

        TrainingTrajectorySelectionOutput {
            selected_metadata,
            all_average_absolute_advantages: self.all_average_absolute_advantages,
            total_trajectories,
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
    old_logprobs: &mut Vec<f32>,
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
    assert_eq!(
        input_ids.len(),
        old_logprobs.len(),
        "input_ids and old_logprobs must have the same length before truncation"
    );
    if input_ids.len() <= MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH {
        return;
    }

    let start = input_ids.len() - MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH;
    input_ids.drain(0..start);
    labels.drain(0..start);
    advantages.drain(0..start);
    old_logprobs.drain(0..start);

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
    assert_eq!(
        input_ids.len(),
        old_logprobs.len(),
        "input_ids and old_logprobs must have the same length after truncation"
    );
    assert!(
        input_ids.len() <= MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH,
        "trajectory length must be capped after truncation"
    );
}

fn clip_training_advantage(value: f32) -> f32 {
    value.clamp(-MAX_ABSOLUTE_TOKEN_ADVANTAGE, MAX_ABSOLUTE_TOKEN_ADVANTAGE)
}

fn generated_token_old_logprob(
    token_id: i32,
    token_logprobs: &crate::llm_model::Top8Candidates,
) -> f32 {
    token_logprobs
        .iter()
        .find(|candidate| candidate.token_id == token_id)
        .unwrap_or_else(|| {
            panic!(
                "generated token id {} missing from stored rollout top-k logprobs",
                token_id
            )
        })
        .logprob
}

#[derive(Debug, Clone, Copy, Default)]
struct AdvantagePolarityTotals {
    positive_token_count: usize,
    negative_token_count: usize,
    total_positive_advantage: f32,
    total_negative_advantage_magnitude: f32,
}

#[derive(Debug, Clone, Copy)]
struct AdvantageBalancingResult {
    pre_balance: AdvantagePolarityTotals,
    post_balance: AdvantagePolarityTotals,
    positive_multiplier: f32,
    negative_multiplier: f32,
}

fn compute_advantage_polarity_totals<M: LlmModelMarker>(
    trajectories: &[DirectTrainingTrajectory<M>],
) -> AdvantagePolarityTotals {
    let mut totals = AdvantagePolarityTotals::default();
    for trajectory in trajectories {
        for advantage in trajectory.advantages.iter().copied() {
            if advantage > 0.0 {
                totals.positive_token_count += 1;
                totals.total_positive_advantage += advantage;
            } else if advantage < 0.0 {
                totals.negative_token_count += 1;
                totals.total_negative_advantage_magnitude += -advantage;
            }
        }
    }
    totals
}

fn compute_advantage_balance_multipliers(totals: AdvantagePolarityTotals) -> (f32, f32) {
    if totals.total_positive_advantage <= ADVANTAGE_BALANCE_EPSILON
        || totals.total_negative_advantage_magnitude <= ADVANTAGE_BALANCE_EPSILON
    {
        return (1.0, 1.0);
    }
    if totals.total_positive_advantage > totals.total_negative_advantage_magnitude {
        (
            totals.total_negative_advantage_magnitude / totals.total_positive_advantage,
            1.0,
        )
    } else {
        (
            1.0,
            totals.total_positive_advantage / totals.total_negative_advantage_magnitude,
        )
    }
}

fn clamp_negative_advantages_to_zero(segment_advantages: &mut BTreeMap<SegmentId, f32>) {
    for advantage in segment_advantages.values_mut() {
        if *advantage < 0.0 {
            *advantage = 0.0;
        }
    }
}

fn balance_training_trajectory_advantages<M: LlmModelMarker>(
    trajectories: &mut [DirectTrainingTrajectory<M>],
) -> AdvantageBalancingResult {
    let pre_balance = compute_advantage_polarity_totals(trajectories);
    let (positive_multiplier, negative_multiplier) =
        compute_advantage_balance_multipliers(pre_balance);
    for trajectory in trajectories.iter_mut() {
        for advantage in trajectory.advantages.iter_mut() {
            if *advantage > 0.0 {
                *advantage *= positive_multiplier;
            } else if *advantage < 0.0 {
                *advantage *= negative_multiplier;
            }
        }
    }
    let post_balance = compute_advantage_polarity_totals(trajectories);
    AdvantageBalancingResult {
        pre_balance,
        post_balance,
        positive_multiplier,
        negative_multiplier,
    }
}

fn count_branch_points<M: LlmModelMarker, S: DatasetSplit>(tree: &DirectTree<'_, M, S>) -> usize {
    tree.segments
        .values()
        .filter(|segment| segment.child_ids.len() > 1)
        .count()
}

fn count_extra_branches<M: LlmModelMarker, S: DatasetSplit>(tree: &DirectTree<'_, M, S>) -> usize {
    tree.segments
        .values()
        .map(|segment| segment.child_ids.len().saturating_sub(1))
        .sum()
}

fn log_extreme_advantage_warning<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    max_absolute_advantage: f32,
) {
    if max_absolute_advantage <= EXTREME_ABSOLUTE_TOKEN_ADVANTAGE_WARNING_THRESHOLD {
        return;
    }
    log_warning(format!(
        "extreme_training_advantage=1 question_flat_id={:?} max_absolute_advantage={:.6} branch_points={} branch_count={} total_segments={}",
        tree.action_log.question.flat_id,
        max_absolute_advantage,
        count_branch_points(tree),
        count_extra_branches(tree),
        tree.segments.len(),
    ));
}

fn supervised_content_advantage_stats<M: LlmModelMarker>(
    segment: &crate::tree::Segment<M>,
    segment_advantage: f32,
) -> Option<(f32, usize)> {
    let mut total_abs = 0.0_f32;
    let mut token_count = 0usize;
    for content in segment.content.iter() {
        if let SegmentContent::ReasoningOrToolCall { tokens, .. } = content {
            assert!(
                segment_advantage.is_finite(),
                "supervised advantage must be finite"
            );
            let clipped = clip_training_advantage(segment_advantage);
            total_abs += clipped.abs() * tokens.tokens.len() as f32;
            token_count += tokens.tokens.len();
        }
    }
    if token_count == 0 {
        None
    } else {
        Some((total_abs, token_count))
    }
}

pub async fn rollout_logs_to_training_trajectories<M: LlmModelMarker>(
    question_map: BTreeMap<usize, HybridDatasetQuestion<Training>>,
    action_store: ActionLogStore<M, Training>,
    rollout_config: RolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    training_trajectories_file_path: String,
    statistics_file_path: String,
    training_advantage_policy: TrainingAdvantagePolicy,
    positive_advantage_only: bool,
    use_tool: bool,
    training_set_sort_mode: TrainingSetSortMode,
) {
    action_store.sort().unwrap();
    let (selection_output, question_map, action_store) =
        select_training_trajectories_from_rollout_logs::<M, Training>(
            question_map,
            action_store,
            rollout_config.clone(),
            posterior_calculation_config.clone(),
            training_advantage_policy,
            positive_advantage_only,
            use_tool,
        )
        .await;
    let TrainingTrajectorySelectionOutput {
        selected_metadata,
        all_average_absolute_advantages,
        total_trajectories,
        average_absolute_advantage_cutoff,
    } = selection_output;
    let mut training_trajectories = materialize_selected_training_trajectories::<M>(
        question_map,
        action_store,
        rollout_config,
        posterior_calculation_config,
        selected_metadata,
        training_advantage_policy,
        positive_advantage_only,
        use_tool,
        training_set_sort_mode,
    )
    .await;
    let adopted_trajectories = training_trajectories.len();
    let advantage_balancing = if positive_advantage_only {
        let totals = compute_advantage_polarity_totals(&training_trajectories);
        AdvantageBalancingResult {
            pre_balance: totals,
            post_balance: totals,
            positive_multiplier: 1.0,
            negative_multiplier: 1.0,
        }
    } else {
        balance_training_trajectory_advantages(&mut training_trajectories)
    };
    write_training_trajectories_msgpack_file(
        &training_trajectories_file_path,
        &training_trajectories,
    )
    .unwrap();

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
        pre_balance_positive_advantage_token_count: advantage_balancing
            .pre_balance
            .positive_token_count,
        pre_balance_negative_advantage_token_count: advantage_balancing
            .pre_balance
            .negative_token_count,
        pre_balance_total_positive_advantage: advantage_balancing
            .pre_balance
            .total_positive_advantage,
        pre_balance_total_negative_advantage_magnitude: advantage_balancing
            .pre_balance
            .total_negative_advantage_magnitude,
        positive_advantage_multiplier: advantage_balancing.positive_multiplier,
        negative_advantage_multiplier: advantage_balancing.negative_multiplier,
        post_balance_total_positive_advantage: advantage_balancing
            .post_balance
            .total_positive_advantage,
        post_balance_total_negative_advantage_magnitude: advantage_balancing
            .post_balance
            .total_negative_advantage_magnitude,
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
    log_key_value_pair(
        "pre_balance_total_positive_advantage",
        statistics.pre_balance_total_positive_advantage.to_string(),
    );
    log_key_value_pair(
        "pre_balance_total_negative_advantage_magnitude",
        statistics
            .pre_balance_total_negative_advantage_magnitude
            .to_string(),
    );
    log_key_value_pair(
        "positive_advantage_multiplier",
        statistics.positive_advantage_multiplier.to_string(),
    );
    log_key_value_pair(
        "negative_advantage_multiplier",
        statistics.negative_advantage_multiplier.to_string(),
    );
    log_key_value_pair(
        "post_balance_total_positive_advantage",
        statistics.post_balance_total_positive_advantage.to_string(),
    );
    log_key_value_pair(
        "post_balance_total_negative_advantage_magnitude",
        statistics
            .post_balance_total_negative_advantage_magnitude
            .to_string(),
    );
    log_info(format!(
        "training_samples_generated={} max_average_absolute_advantage={} min_average_absolute_advantage={} median_average_absolute_advantage={} pre_balance_positive_total={} pre_balance_negative_total={} positive_multiplier={} negative_multiplier={} post_balance_positive_total={} post_balance_negative_total={}",
        statistics.adopted_trajectories,
        statistics.max_average_absolute_advantage,
        statistics.min_average_absolute_advantage,
        median_average_absolute_advantage,
        statistics.pre_balance_total_positive_advantage,
        statistics.pre_balance_total_negative_advantage_magnitude,
        statistics.positive_advantage_multiplier,
        statistics.negative_advantage_multiplier,
        statistics.post_balance_total_positive_advantage,
        statistics.post_balance_total_negative_advantage_magnitude,
    ));
}

async fn select_training_trajectories_from_rollout_logs<M: LlmModelMarker, S: DatasetSplit>(
    question_map: BTreeMap<usize, HybridDatasetQuestion<S>>,
    action_store: ActionLogStore<M, S>,
    rollout_config: RolloutConfig<S>,
    posterior_calculation_config: PosteriorCalculationConfig,
    training_advantage_policy: TrainingAdvantagePolicy,
    positive_advantage_only: bool,
    use_tool: bool,
) -> (
    TrainingTrajectorySelectionOutput<S>,
    BTreeMap<usize, HybridDatasetQuestion<S>>,
    ActionLogStore<M, S>,
) {
    // let mut keys = action_log_store.metadata_store.get_keys().unwrap();
    let mut keys = action_store.get_keys().unwrap();
    let num_keys = keys.len();
    keys.sort();

    let mut selection_state = TrajectorySelectionState::new(num_keys);
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
                let question = question_map.get(&key.0).unwrap().clone();
                let actions = action_store.load_action_log(key).unwrap();
                let action_log = DirectTreeActionLog {
                    mount_dir: String::new(),
                    config_nickname: String::new(),
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    use_tool,
                    fixed_temperature: constants::temperature_by_split::<S>(),
                    actions,
                };
                let training_advantage_policy = training_advantage_policy.clone();
                join_set.spawn(async move {
                    let _permit = permit;
                    let trajectory_summaries = action_log_to_candidate_summaries::<M, S>(
                        action_log,
                        training_advantage_policy,
                        positive_advantage_only,
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

    (selection_state.into_output(), question_map, action_store)
}

async fn materialize_selected_training_trajectories<M: LlmModelMarker>(
    question_map: BTreeMap<usize, HybridDatasetQuestion<Training>>,
    action_store: ActionLogStore<M, Training>,
    rollout_config: RolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    mut selected_metadata: Vec<TrajectoryMetadata<Training>>,
    training_advantage_policy: TrainingAdvantagePolicy,
    positive_advantage_only: bool,
    use_tool: bool,
    training_set_sort_mode: TrainingSetSortMode,
) -> Vec<DirectTrainingTrajectory<M>> {
    match training_set_sort_mode {
        TrainingSetSortMode::ByLengthAscending => {
            selected_metadata.sort_by(|a, b| {
                a.trajectory_token_length
                    .cmp(&b.trajectory_token_length)
                    .then_with(|| {
                        b.average_absolute_advantage
                            .cmp(&a.average_absolute_advantage)
                    })
                    .then_with(|| a.question_flat_id.cmp(&b.question_flat_id))
                    .then_with(|| a.trajectory_index.cmp(&b.trajectory_index))
            });
        }
        TrainingSetSortMode::ByLengthDescending => {
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
        }
        TrainingSetSortMode::ByQuestion => {
            selected_metadata.sort_by(|a, b| {
                a.question_flat_id
                    .cmp(&b.question_flat_id)
                    .then_with(|| a.trajectory_index.cmp(&b.trajectory_index))
            });
        }
        TrainingSetSortMode::RandomShuffle => {
            let mut rng = rand::rng();
            selected_metadata.shuffle(&mut rng);
        }
    }

    let adopted_trajectories = selected_metadata.len();
    assert!(
        adopted_trajectories > 0,
        "trajectory selection kept zero trajectories; check advantage distribution"
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
    let mut outputs_by_index: Vec<Option<DirectTrainingTrajectory<M>>> =
        vec![None; adopted_trajectories];
    while !grouped_metadata.is_empty() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if !grouped_metadata.is_empty() => {
                let permit = permit_result.expect("materialization semaphore should not be closed");
                let (question_flat_id, selected_for_question) = grouped_metadata
                    .pop_first()
                    .expect("grouped metadata should be non-empty");
                let key = question_flat_id;
                let question = question_map.get(&key.0).unwrap().clone();
                let actions = action_store.load_action_log(key).unwrap();
                let action_log = DirectTreeActionLog{
                    mount_dir: String::new(),
                    config_nickname: String::new(),
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    use_tool,
                    fixed_temperature: NotNan::new(constants::TRAINING_TEMPERATURE).unwrap(),
                    actions,
                };
                let training_advantage_policy = training_advantage_policy.clone();
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
                        training_advantage_policy,
                        &selected_trajectory_indices,
                        positive_advantage_only,
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
                            assert!(
                                output_index < outputs_by_index.len(),
                                "trajectory output index {} out of bounds for {} selected trajectories",
                                output_index,
                                outputs_by_index.len(),
                            );
                            outputs_by_index[output_index] = Some(trajectory);
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

    outputs_by_index
        .into_iter()
        .enumerate()
        .map(|(output_index, trajectory)| {
            trajectory.unwrap_or_else(|| {
                panic!("trajectory output {} was not materialized", output_index)
            })
        })
        .collect()
}

pub fn write_training_trajectories_msgpack_file<M: LlmModelMarker>(
    file_path: &str,
    trajectories: &[DirectTrainingTrajectory<M>],
) -> Result<(), String> {
    let path = std::path::Path::new(file_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create training trajectories parent dir {}: {}",
                parent.display(),
                err
            )
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| {
            format!(
                "Failed to create training trajectories msgpack file {}: {}",
                path.display(),
                err
            )
        })?;
    for trajectory in trajectories {
        let bytes = rmp_serde::to_vec_named(trajectory).map_err(|err| {
            format!(
                "Failed to serialize training trajectory for msgpack file {}: {}",
                path.display(),
                err
            )
        })?;
        file.write_all(&bytes).map_err(|err| {
            format!(
                "Failed to append training trajectory to msgpack file {}: {}",
                path.display(),
                err
            )
        })?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TrainingTrajectoryConfigBundle<S: DatasetSplit> {
    pub rollout_config: RolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectTrainingSetStatistics {
    pub average_absolute_advantages_sorted: Vec<f32>, // sorted from high to low
    pub max_average_absolute_advantage: f32,
    pub min_average_absolute_advantage: f32,
    pub average_absolute_advantage_cutoff: f32,
    pub total_trajectories: usize,
    pub adopted_trajectories: usize,
    #[serde(default)]
    pub pre_balance_positive_advantage_token_count: usize,
    #[serde(default)]
    pub pre_balance_negative_advantage_token_count: usize,
    #[serde(default)]
    pub pre_balance_total_positive_advantage: f32,
    #[serde(default)]
    pub pre_balance_total_negative_advantage_magnitude: f32,
    #[serde(default = "default_advantage_multiplier")]
    pub positive_advantage_multiplier: f32,
    #[serde(default = "default_advantage_multiplier")]
    pub negative_advantage_multiplier: f32,
    #[serde(default)]
    pub post_balance_total_positive_advantage: f32,
    #[serde(default)]
    pub post_balance_total_negative_advantage_magnitude: f32,
}

fn default_advantage_multiplier() -> f32 {
    1.0
}

fn initialize_training_segment_advantages<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    training_advantage_policy: TrainingAdvantagePolicy,
) -> BTreeMap<SegmentId, f32> {
    let mut segment_advantages = match training_advantage_policy {
        TrainingAdvantagePolicy::TreeMappoPosterior => {
            tree.calculate_segment_advantages_from_posteriors(None)
        }
        TrainingAdvantagePolicy::TreeRpoWinRate => {
            tree.calculate_segment_advantages_from_win_rate()
        }
        TrainingAdvantagePolicy::GrpoTerminalReward => {
            tree.calculate_segment_advantages_from_grpo_terminal_reward()
        }
    };
    for segment_id in tree.segments.keys().copied() {
        segment_advantages.entry(segment_id).or_insert(0.0);
    }
    segment_advantages
}

fn trajectory_average_absolute_advantage<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    segment_ids: &[SegmentId],
    segment_advantages: &BTreeMap<SegmentId, f32>,
) -> f32 {
    let mut total_abs_advantage = 0.0_f32;
    let mut supervised_token_count = 0usize;
    for segment_id in segment_ids.iter() {
        let segment = tree.segments.get(segment_id).unwrap();
        let segment_advantage = *segment_advantages.get(segment_id).unwrap();
        if let Some((segment_abs_sum, segment_token_count)) =
            supervised_content_advantage_stats(segment, segment_advantage)
        {
            total_abs_advantage += segment_abs_sum;
            supervised_token_count += segment_token_count;
        }
    }
    assert!(
        supervised_token_count > 0,
        "trajectory must contain supervised tokens"
    );
    total_abs_advantage / supervised_token_count as f32
}

fn trajectory_token_length<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    segment_ids: &[SegmentId],
) -> usize {
    let mut trajectory_token_length = 0usize;
    for segment_id in segment_ids.iter() {
        let segment = tree.segments.get(segment_id).unwrap();
        for content in segment.content.iter() {
            let token_count = match content {
                SegmentContent::Prompt(token_array) | SegmentContent::ToolResponse(token_array) => {
                    token_array.tokens.len()
                }
                SegmentContent::ReasoningOrToolCall {
                    tokens,
                    complete: _,
                } => tokens.tokens.len(),
            };
            trajectory_token_length += token_count;
        }
    }
    trajectory_token_length
}

fn best_leaf_by_average_absolute_advantage<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<'_, M, S>,
    leaf_segment_ids: &BTreeSet<SegmentId>,
    segment_advantages: &BTreeMap<SegmentId, f32>,
) -> (SegmentId, f32, Vec<SegmentId>) {
    leaf_segment_ids
        .iter()
        .map(|leaf| {
            let segment_ids = tree.get_trajectory_segments_till_id(*leaf);
            let average_absolute_advantage =
                trajectory_average_absolute_advantage(tree, &segment_ids, segment_advantages);
            (*leaf, average_absolute_advantage, segment_ids)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

fn zero_taken_segment_advantages(
    segment_advantages: &mut BTreeMap<SegmentId, f32>,
    segment_ids: &[SegmentId],
) {
    for segment_id in segment_ids.iter().copied() {
        segment_advantages.insert(segment_id, 0.0);
    }
}

fn action_log_to_candidate_summaries<M: LlmModelMarker, S: DatasetSplit>(
    action_log: DirectTreeActionLog<M, S>,
    training_advantage_policy: TrainingAdvantagePolicy,
    positive_advantage_only: bool,
) -> Vec<TrajectorySummary<S>> {
    let tree = DirectTree::from_action_log(&action_log);
    if !ALLOW_INCOMPLETE && !tree.completed() {
        return Vec::new();
    }
    if !matches!(tree.get_correctness(), TreeCorrectness::Mixed) {
        return Vec::new();
    }
    let mut segment_advantages =
        initialize_training_segment_advantages(&tree, training_advantage_policy);
    if positive_advantage_only {
        clamp_negative_advantages_to_zero(&mut segment_advantages);
    }
    let max_absolute_advantage = segment_advantages
        .values()
        .map(|advantage| advantage.abs())
        .fold(0.0_f32, f32::max);
    log_extreme_advantage_warning(&tree, max_absolute_advantage);

    let mut trajectory_summaries: Vec<TrajectorySummary<S>> = Vec::new();
    let mut leaf_segment_ids: BTreeSet<SegmentId> =
        tree.leaf_segment_judgments.keys().cloned().collect();
    while !leaf_segment_ids.is_empty() {
        let (best_leaf, average_absolute_advantage, segment_ids) =
            best_leaf_by_average_absolute_advantage(&tree, &leaf_segment_ids, &segment_advantages);
        trajectory_summaries.push(TrajectorySummary {
            question_flat_id: tree.action_log.question.flat_id,
            leaf_segment_id: best_leaf,
            average_absolute_advantage,
            trajectory_token_length: trajectory_token_length(&tree, &segment_ids)
                .min(MAX_TRAINING_TRAJECTORY_TOKEN_LENGTH),
        });
        zero_taken_segment_advantages(&mut segment_advantages, &segment_ids);
        leaf_segment_ids.remove(&best_leaf);
    }
    trajectory_summaries
}

fn action_log_to_selected_trajectories<M: LlmModelMarker>(
    action_log: DirectTreeActionLog<M, Training>,
    training_advantage_policy: TrainingAdvantagePolicy,
    selected_trajectory_indices: &BTreeSet<usize>,
    positive_advantage_only: bool,
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
    let mut segment_advantages =
        initialize_training_segment_advantages(&tree, training_advantage_policy);
    if positive_advantage_only {
        clamp_negative_advantages_to_zero(&mut segment_advantages);
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
        let (best_leaf, average_absolute_advantage, segment_ids) =
            best_leaf_by_average_absolute_advantage(&tree, &leaf_segment_ids, &segment_advantages);
        let should_materialize = selected_trajectory_indices.contains(&trajectory_index);

        if should_materialize {
            let mut input_ids: Vec<i32> = Vec::new();
            let mut labels: Vec<i32> = Vec::new();
            let mut advantages: Vec<f32> = Vec::new();
            let mut old_logprobs: Vec<f32> = Vec::new();
            for segment_id in segment_ids.iter() {
                let segment = tree.segments.get(segment_id).unwrap();
                let segment_advantage = *segment_advantages.get(segment_id).unwrap();
                for content in segment.content.iter() {
                    match content {
                        SegmentContent::Prompt(token_array)
                        | SegmentContent::ToolResponse(token_array) => {
                            input_ids.extend(token_array.tokens.iter());
                            labels.extend(vec![-100; token_array.tokens.len()]); // we set the labels for the prompt tokens to -100 so that they will be ignored in the loss calculation
                            advantages.extend(vec![0.0; token_array.tokens.len()]);
                            old_logprobs.extend(vec![0.0; token_array.tokens.len()]);
                        }
                        SegmentContent::ReasoningOrToolCall {
                            tokens,
                            complete: _,
                        } => {
                            assert!(
                                segment_advantage.is_finite(),
                                "supervised advantage must be finite"
                            );
                            let clipped_advantage = clip_training_advantage(segment_advantage);
                            input_ids.extend(tokens.tokens.iter());
                            labels.extend(tokens.tokens.iter());
                            advantages.extend(vec![clipped_advantage; tokens.tokens.len()]);
                            old_logprobs.extend(
                                tokens.tokens.iter().zip(tokens.logprobs.iter()).map(
                                    |(token_id, token_logprobs)| {
                                        generated_token_old_logprob(*token_id, token_logprobs)
                                    },
                                ),
                            );
                        }
                    }
                }
            }
            truncate_trajectory_tokens(
                &mut input_ids,
                &mut labels,
                &mut advantages,
                &mut old_logprobs,
            );
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
                    old_logprobs,
                    average_absolute_segment_advantage: average_absolute_advantage,
                    _phantom: std::marker::PhantomData::<M>,
                },
            ));
        }
        zero_taken_segment_advantages(&mut segment_advantages, &segment_ids);
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
    pub old_logprobs: Vec<f32>,
    pub average_absolute_segment_advantage: f32,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<M>,
}

pub fn training_trajectories_file_path<M: LlmModelMarker>(
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    training_trajectories_path(mount_dir, M::CLI_NAME, config_nickname, epoch)
}

pub fn training_trajectories_msgpack_file_path<M: LlmModelMarker>(
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    Path::new(&training_trajectories_file_path::<M>(
        mount_dir,
        config_nickname,
        epoch,
    ))
    .join("trajectories.msgpack")
    .to_string_lossy()
    .into_owned()
}

pub fn training_trajectories_config_bundle_file_path<M: LlmModelMarker>(
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    Path::new(&training_trajectories_file_path::<M>(
        mount_dir,
        config_nickname,
        epoch,
    ))
    .join("config_bundle.json")
    .to_string_lossy()
    .into_owned()
}

pub fn training_trajectories_stats_file_path<M: LlmModelMarker>(
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    training_trajectories_stats_path(mount_dir, M::CLI_NAME, config_nickname, epoch)
}

pub fn open_training_trajectories<M: LlmModelMarker>(
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
) -> Vec<DirectTrainingTrajectory<M>> {
    let training_trajectories_path =
        training_trajectories_file_path::<M>(mount_dir, config_nickname, epoch);
    let file_path = if Path::new(&training_trajectories_path).is_dir() {
        training_trajectories_msgpack_file_path::<M>(mount_dir, config_nickname, epoch)
    } else {
        training_trajectories_path
    };
    let file = File::open(&file_path).unwrap_or_else(|e| {
        panic!(
            "Failed to open training trajectories msgpack file at {}: {}",
            file_path, e
        )
    });
    let mut bytes = Vec::new();
    BufReader::new(file)
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to read training trajectories msgpack file at {}: {}",
                file_path, e
            )
        });
    let mut cursor = Cursor::new(bytes.as_slice());
    let mut trajectories = Vec::new();
    let total_len = bytes.len() as u64;
    while cursor.position() < total_len {
        let mut deserializer = rmp_serde::Deserializer::new(&mut cursor);
        let trajectory = DirectTrainingTrajectory::<M>::deserialize(&mut deserializer)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to deserialize training trajectory from msgpack file at {}: {}",
                    file_path, e
                )
            });
        trajectories.push(trajectory);
    }
    trajectories
}

pub async fn generate_training_trajectories<M: LlmModelMarker>(
    mount_dir: &str,
    config_nickname: &str,
    rollout_config: RolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize,
    training_advantage_policy: TrainingAdvantagePolicy,
    positive_advantage_only: bool,
    use_tool: bool,
    training_set_sort_mode: TrainingSetSortMode,
) {
    let training_trajectories_path =
        training_trajectories_file_path::<M>(mount_dir, config_nickname, epoch);
    let training_trajectories_path = Path::new(&training_trajectories_path);
    std::fs::create_dir_all(training_trajectories_path).unwrap();
    let file_path = training_trajectories_msgpack_file_path::<M>(mount_dir, config_nickname, epoch);
    if Path::new(&file_path).exists() {
        std::fs::remove_file(&file_path).unwrap();
    }
    let stats_file_path =
        training_trajectories_stats_file_path::<M>(mount_dir, config_nickname, epoch);
    let config_bundle_path =
        training_trajectories_config_bundle_file_path::<M>(mount_dir, config_nickname, epoch);
    write_json(
        &config_bundle_path,
        &TrainingTrajectoryConfigBundle {
            rollout_config: rollout_config.clone(),
            posterior_calculation_config: posterior_calculation_config.clone(),
        },
    )
    .unwrap();
    let dataset_store = open_hybrid_dataset::<Training>();
    let question_map: BTreeMap<usize, HybridDatasetQuestion<Training>> = dataset_store
        .iter()
        .expect("failed to iterate hybrid dataset")
        .map(|r| r.expect("failed to read question from hybrid dataset"))
        .collect();
    let action_store = open_action_logs::<M, Training>(mount_dir, config_nickname, epoch);
    action_store
        .write_config_bundle_if_missing(&ActionLogConfigBundle {
            rollout_config: rollout_config.clone(),
            posterior_calculation_config: posterior_calculation_config.clone(),
            use_tool,
            fixed_temperature: NotNan::new(constants::TRAINING_TEMPERATURE).unwrap(),
        })
        .unwrap();
    rollout_logs_to_training_trajectories::<M>(
        question_map,
        action_store,
        rollout_config,
        posterior_calculation_config,
        file_path,
        stats_file_path,
        training_advantage_policy,
        positive_advantage_only,
        use_tool,
        training_set_sort_mode,
    )
    .await;
    log_info("Finished generating training trajectories file.");
}

/// Generate training trajectories using custom action log store path.
/// Used by the one-shot training program where action logs live at a non-standard path.
pub async fn generate_training_trajectories_with_path<M: LlmModelMarker>(
    action_log_store_path: &str,
    training_trajectories_dir: &str,
    training_trajectories_msgpack_path: &str,
    stats_file_path: &str,
    config_bundle_path: &str,
    rollout_config: RolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    training_advantage_policy: TrainingAdvantagePolicy,
    positive_advantage_only: bool,
    use_tool: bool,
    training_set_sort_mode: TrainingSetSortMode,
) {
    std::fs::create_dir_all(training_trajectories_dir).unwrap();
    if Path::new(training_trajectories_msgpack_path).exists() {
        std::fs::remove_file(training_trajectories_msgpack_path).unwrap();
    }
    write_json(
        config_bundle_path,
        &TrainingTrajectoryConfigBundle {
            rollout_config: rollout_config.clone(),
            posterior_calculation_config: posterior_calculation_config.clone(),
        },
    )
    .unwrap();
    let dataset_store = open_hybrid_dataset::<Training>();
    let question_map: BTreeMap<usize, HybridDatasetQuestion<Training>> = dataset_store
        .iter()
        .expect("failed to iterate hybrid dataset")
        .map(|r| r.expect("failed to read question from hybrid dataset"))
        .collect();
    let action_store =
        ActionLogStore::<M, Training>::initialize_if_missing(action_log_store_path.to_string())
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to open action log store at {}: {}",
                    action_log_store_path, e
                )
            });
    action_store
        .write_config_bundle_if_missing(&ActionLogConfigBundle {
            rollout_config: rollout_config.clone(),
            posterior_calculation_config: posterior_calculation_config.clone(),
            use_tool,
            fixed_temperature: NotNan::new(constants::TRAINING_TEMPERATURE).unwrap(),
        })
        .unwrap();
    rollout_logs_to_training_trajectories::<M>(
        question_map,
        action_store,
        rollout_config,
        posterior_calculation_config,
        training_trajectories_msgpack_path.to_string(),
        stats_file_path.to_string(),
        training_advantage_policy,
        positive_advantage_only,
        use_tool,
        training_set_sort_mode,
    )
    .await;
    log_info("Finished generating one-shot training trajectories.");
}
