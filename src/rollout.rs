use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::marker::PhantomData;

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use arc_swap::ArcSwapOption;
use serde::{Deserialize, Serialize};

use kll_rs::KllFloatSketch;
use reqwest::Client;
use research_utility::progress_text_logger::{
    delete_worker_progress_bar, log_info, log_key_value_pair, log_master_progress, log_warning,
    log_worker_progress,
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use ordered_float::NotNan;

use crate::{
    atomic_count_guard::AtomicCountGuardRef,
    directories::{tree_artifacts_oneshot_chunk_done_path, tree_artifacts_oneshot_chunk_path},
    hybrid_dataset::{DatasetSplit, HybridDatasetQuestion, QuestionFlatId, open_hybrid_dataset},
    llm_model::{InferenceEndpoint, LlmCallable, LlmModelMarker},
    model_answer_judgment_cache::{
        commit_pending_writes_if_any, reset_model_answer_judgment_cache_if_any,
    },
    posterior_calculation_config::PosteriorCalculationConfig,
    rollout_config::RolloutConfig,
    tool_call_python::init_python_tool_pool,
    trajectory::{FailureMode, FinalAnswer},
    tree::{DirectTree, SegmentContent, TreeCorrectness},
    tree_action::DirectTreeAction,
    tree_action_log::{
        ActionLogConfigBundle, ActionLogStore, DirectTreeActionLog, open_action_logs,
    },
    tree_artifact::{TreeArtifact, write_tree_artifacts_msgpack},
    tree_status::{
        DirectTreeStatus, GuidedBranchingSubStatus, SpontaneousBranchingSubStatus, TrunkSubStatus,
    },
    tree_to_action::BranchingRuntimeOptions,
};

pub(crate) struct DistributionStats {
    sketch: KllFloatSketch,
    num_samples: usize,
    min: usize,
    max: usize,
}

impl DistributionStats {
    fn new() -> Self {
        Self {
            sketch: KllFloatSketch::new().expect("failed to initialize KLL sketch"),
            num_samples: 0,
            min: 0,
            max: 0,
        }
    }

    fn update_and_get_summary(&mut self, value: usize) -> DistributionSummary {
        self.sketch.update(value as f32);
        if self.num_samples == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = std::cmp::min(self.min, value);
            self.max = std::cmp::max(self.max, value);
        }
        self.num_samples += 1;

        DistributionSummary {
            min: self.min,
            median: quantile_to_usize(self.sketch.get_quantile(0.5)),
            q3: quantile_to_usize(self.sketch.get_quantile(0.75)),
            max: self.max,
        }
    }
}

struct DistributionSummary {
    min: usize,
    median: usize,
    q3: usize,
    max: usize,
}

fn condensed_distribution(summary: &DistributionSummary) -> String {
    format!(
        "({}, {}, {}, {})",
        summary.min, summary.median, summary.q3, summary.max
    )
}

fn quantile_to_usize(value: f32) -> usize {
    if !value.is_finite() {
        return 0;
    }
    value.round().max(0.0) as usize
}

fn action_is_llm_call<M: LlmModelMarker>(action: &DirectTreeAction<M>) -> bool {
    matches!(
        action,
        DirectTreeAction::AppendSegmentContent(SegmentContent::ReasoningOrToolCall { .. })
    )
}

fn trajectory_length_being_judged<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<M, S>,
) -> Option<usize> {
    match &tree.status {
        DirectTreeStatus::WorkingOnTrunk(TrunkSubStatus::JudgingSegment {
            finalized_content_array,
            ..
        }) => {
            let root_segment_id = tree.root_segment_id.expect("root segment must exist");
            Some(
                tree.get_trajectory(root_segment_id, finalized_content_array)
                    .to_prompt_tokens()
                    .len(),
            )
        }
        DirectTreeStatus::WorkingOnGuidedBranching(GuidedBranchingSubStatus::JudgingSegment {
            parent_segment_id,
            finalized_content_array,
            ..
        }) => Some(
            tree.get_trajectory(*parent_segment_id, finalized_content_array)
                .to_prompt_tokens()
                .len(),
        ),
        DirectTreeStatus::WorkingOnSpontaneousBranching(
            SpontaneousBranchingSubStatus::JudgingSegment {
                parent_segment_id,
                prefix_trimmed_content_array,
                ..
            },
        ) => Some(
            tree.get_trajectory(*parent_segment_id, prefix_trimmed_content_array)
                .to_prompt_tokens()
                .len(),
        ),
        _ => None,
    }
}

async fn run_progress_timer(
    start_time: Instant,
    deadline: Instant,
    total_secs: f32,
    mount_dir: String,
    model_cli_name: String,
    config_nickname: String,
) {
    let rollout_stats = RolloutStats::global();
    let log_time_progress = |now: Instant| {
        let elapsed_secs = (now - start_time).as_secs_f32().min(total_secs);
        let progress = (elapsed_secs / total_secs).min(1.0);
        let label = format!("Rollout: ({elapsed_secs:.1}s/{total_secs:.1}s)");
        log_master_progress(progress, &label);
    };

    let progress_log_interval = Duration::from_secs(1);
    let mut next_progress_log_time = start_time;
    let mut next_throughput_log_time = start_time;
    let throughput_window = Duration::from_secs(5);
    let throughput_log_interval = Duration::from_millis(500);
    let database_commit_interval = Duration::from_secs(20 * 60);
    let mut next_database_commit_request_time = start_time + database_commit_interval;
    let model_answer_judgment_cache_commit_interval = Duration::from_secs(5 * 60);
    let mut next_model_answer_judgment_cache_commit_time =
        start_time + model_answer_judgment_cache_commit_interval;
    let segment_count = 8usize;
    let mut next_segment_to_log = 1usize;
    let mut llm_call_samples: VecDeque<(Instant, usize)> = VecDeque::new();
    loop {
        if ROLLOUT_STOP_SIGNAL.load(Ordering::Relaxed) {
            break;
        }
        let now = Instant::now();
        if now >= next_progress_log_time {
            log_time_progress(now);
            // If the runtime was blocked for a while, avoid "catch-up" bursts by
            // scheduling from the current time rather than replaying missed ticks.
            next_progress_log_time = now + progress_log_interval;
        }

        while next_segment_to_log <= segment_count {
            let segment_elapsed_secs =
                total_secs * (next_segment_to_log as f32) / segment_count as f32;
            let segment_deadline = start_time + Duration::from_secs_f32(segment_elapsed_secs);
            if now < segment_deadline {
                break;
            }
            let elapsed_secs = (now - start_time).as_secs_f32().min(total_secs);
            let finished_trees = rollout_stats.num_finished_trees.load(Ordering::Relaxed);
            let finished_branches = rollout_stats.num_finished_branches.load(Ordering::Relaxed);
            log_info(format!(
                "Rollout time segment {}/{} reached: elapsed={elapsed_secs:.1}s/{total_secs:.1}s, finished_trees={}, finished_branches={}",
                next_segment_to_log, segment_count, finished_trees, finished_branches,
            ));
            next_segment_to_log += 1;
        }

        if now >= next_database_commit_request_time {
            rollout_stats
                .database_commit_after_next_write
                .store(true, Ordering::Relaxed);
            next_database_commit_request_time = now + database_commit_interval;
        }

        if now >= next_model_answer_judgment_cache_commit_time {
            if let Err(error) =
                commit_pending_writes_if_any(&mount_dir, &model_cli_name, &config_nickname)
            {
                log_warning(format!(
                    "Failed to commit model answer judgment cache during rollout: {}",
                    error
                ));
            }
            next_model_answer_judgment_cache_commit_time =
                now + model_answer_judgment_cache_commit_interval;
        }

        if now >= next_throughput_log_time {
            let current_llm_calls = rollout_stats.total_llm_calls.load(Ordering::Relaxed);
            llm_call_samples.push_back((now, current_llm_calls));
            while let Some((sample_time, _)) = llm_call_samples.front() {
                if now.duration_since(*sample_time) > throughput_window {
                    llm_call_samples.pop_front();
                } else {
                    break;
                }
            }

            let llm_call_throughput =
                if let Some((oldest_time, oldest_count)) = llm_call_samples.front() {
                    let elapsed_secs = now.duration_since(*oldest_time).as_secs_f64();
                    if elapsed_secs <= f64::EPSILON {
                        0.0
                    } else {
                        (current_llm_calls.saturating_sub(*oldest_count)) as f64 / elapsed_secs
                    }
                } else {
                    0.0
                };

            log_key_value_pair(
                "llm_call_throughput_per_sec_5s_window",
                format!("{llm_call_throughput:.2}"),
            );
            // Same strategy as progress logging: skip missed intervals.
            next_throughput_log_time = now + throughput_log_interval;
        }

        if now >= deadline {
            ROLLOUT_STOP_SIGNAL.store(true, Ordering::Relaxed);
            break;
        }

        let wake_time = std::cmp::min(
            std::cmp::min(
                std::cmp::min(
                    next_progress_log_time,
                    std::cmp::min(next_throughput_log_time, next_database_commit_request_time),
                ),
                std::cmp::min(next_model_answer_judgment_cache_commit_time, deadline),
            ),
            std::cmp::min(deadline, Instant::now() + Duration::from_millis(100)),
        );
        sleep_until(wake_time).await;
    }

    log_time_progress(Instant::now());
}

static ROLLOUT_STATS: ArcSwapOption<RolloutStats> = ArcSwapOption::const_empty();
pub(crate) static ROLLOUT_STOP_SIGNAL: AtomicBool = AtomicBool::new(false);

pub struct RolloutStats {
    pub(crate) sglang_waiting_workers: AtomicUsize,
    pub(crate) judge_waiting_workers: AtomicUsize,
    pub(crate) tool_waiting_workers: AtomicUsize,
    pub(crate) database_waiting_workers: AtomicUsize,
    pub(crate) num_finished_branches: AtomicUsize,
    pub(crate) num_finished_trees: AtomicUsize,
    pub(crate) num_correct_branches: AtomicUsize,
    pub(crate) num_all_correct_trees: AtomicUsize,
    pub(crate) num_all_incorrect_trees: AtomicUsize,
    pub(crate) num_unjudged_trees: AtomicUsize,
    pub(crate) model_answers_completed: AtomicUsize,
    pub(crate) model_answers_context_window_overflow: AtomicUsize,
    pub(crate) model_answers_only_eos: AtomicUsize,
    pub(crate) model_answers_too_many_turns: AtomicUsize,
    pub(crate) cache_hit_attempts: AtomicUsize,
    pub(crate) cache_hit_count: AtomicUsize,
    pub(crate) database_commit_after_next_write: AtomicBool,
    pub(crate) database_commit_count: AtomicUsize,
    pub(crate) llm_call_stats: tokio::sync::Mutex<DistributionStats>,
    pub(crate) trajectory_length_stats: tokio::sync::Mutex<DistributionStats>,
    pub(crate) correct_trajectory_length_stats: tokio::sync::Mutex<DistributionStats>,
    pub(crate) num_active_rollouts: AtomicUsize,
    pub(crate) total_llm_calls: AtomicUsize,
    pub(crate) tool_calls_processed: AtomicUsize,
    pub(crate) trajectories_per_tree: usize,
    pub(crate) total_trees_to_finish: usize,
}

struct RolloutAllGuard;

impl Drop for RolloutAllGuard {
    fn drop(&mut self) {
        ROLLOUT_STOP_SIGNAL.store(true, Ordering::Relaxed);
        ROLLOUT_STATS.store(None);
    }
}

impl RolloutAllGuard {
    fn new(trajectories_per_tree: usize, total_trees_to_finish: usize) -> Self {
        ROLLOUT_STOP_SIGNAL.store(false, Ordering::Relaxed);
        let stats = Arc::new(RolloutStats {
            sglang_waiting_workers: AtomicUsize::new(0),
            judge_waiting_workers: AtomicUsize::new(0),
            tool_waiting_workers: AtomicUsize::new(0),
            database_waiting_workers: AtomicUsize::new(0),
            num_finished_branches: AtomicUsize::new(0),
            num_finished_trees: AtomicUsize::new(0),
            num_correct_branches: AtomicUsize::new(0),
            num_all_correct_trees: AtomicUsize::new(0),
            num_all_incorrect_trees: AtomicUsize::new(0),
            num_unjudged_trees: AtomicUsize::new(0),
            model_answers_completed: AtomicUsize::new(0),
            model_answers_context_window_overflow: AtomicUsize::new(0),
            model_answers_only_eos: AtomicUsize::new(0),
            model_answers_too_many_turns: AtomicUsize::new(0),
            cache_hit_attempts: AtomicUsize::new(0),
            cache_hit_count: AtomicUsize::new(0),
            database_commit_after_next_write: AtomicBool::new(false),
            database_commit_count: AtomicUsize::new(0),
            llm_call_stats: tokio::sync::Mutex::new(DistributionStats::new()),
            trajectory_length_stats: tokio::sync::Mutex::new(DistributionStats::new()),
            correct_trajectory_length_stats: tokio::sync::Mutex::new(DistributionStats::new()),
            num_active_rollouts: AtomicUsize::new(0),
            total_llm_calls: AtomicUsize::new(0),
            tool_calls_processed: AtomicUsize::new(0),
            trajectories_per_tree,
            total_trees_to_finish,
        });
        ROLLOUT_STATS.store(Some(stats));
        log_key_value_pair("database_commit_count", "0");
        RolloutAllGuard
    }
}

impl RolloutStats {
    pub fn global() -> Arc<Self> {
        ROLLOUT_STATS
            .load_full()
            .expect("rollout stats must be initialized before use")
    }

    fn log_model_answer_counts(&self) {
        log_key_value_pair(
            "model_answers (completed, context, eos, turns)",
            format!(
                "({}, {}, {}, {})",
                self.model_answers_completed.load(Ordering::Relaxed),
                self.model_answers_context_window_overflow
                    .load(Ordering::Relaxed),
                self.model_answers_only_eos.load(Ordering::Relaxed),
                self.model_answers_too_many_turns.load(Ordering::Relaxed)
            ),
        );
    }

    pub(crate) fn reset_model_answer_judgment_cache_hit_rate(&self) {
        self.cache_hit_attempts.store(0, Ordering::Relaxed);
        self.cache_hit_count.store(0, Ordering::Relaxed);
        log_key_value_pair("cache_hit_rate", "0.000000".to_string());
    }

    pub(crate) fn record_model_answer_judgment_cache_read_attempt(&self, cache_hit: bool) {
        let attempts = self.cache_hit_attempts.fetch_add(1, Ordering::Relaxed) + 1;
        let hits = if cache_hit {
            self.cache_hit_count.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            self.cache_hit_count.load(Ordering::Relaxed)
        };
        let cache_hit_rate = if attempts == 0 {
            0.0
        } else {
            hits as f32 / attempts as f32
        };
        log_key_value_pair("cache_hit_rate", format!("{cache_hit_rate:.6}"));
    }

    async fn log_trajectory_length(&self, trajectory_length: usize, is_correct: bool) {
        let summary = self
            .trajectory_length_stats
            .lock()
            .await
            .update_and_get_summary(trajectory_length);
        log_key_value_pair(
            "trajectory_length (min, median, q3, max)",
            condensed_distribution(&summary),
        );
        if is_correct {
            let correct_summary = self
                .correct_trajectory_length_stats
                .lock()
                .await
                .update_and_get_summary(trajectory_length);
            log_key_value_pair(
                "correct_trajectory_length (min, median, q3, max)",
                condensed_distribution(&correct_summary),
            );
        }
    }

    async fn log_llm_calls_per_tree(&self, llm_calls_so_far: usize) {
        let summary = self
            .llm_call_stats
            .lock()
            .await
            .update_and_get_summary(llm_calls_so_far);
        log_key_value_pair(
            "llm_calls_per_tree (min, median, q3, max)",
            condensed_distribution(&summary),
        );
    }

    fn log_tree_completion(&self, tree_correctness: TreeCorrectness, has_judgments: bool) {
        if has_judgments {
            match tree_correctness {
                TreeCorrectness::AllCorrect => {
                    self.num_all_correct_trees.fetch_add(1, Ordering::Relaxed);
                }
                TreeCorrectness::AllIncorrect => {
                    self.num_all_incorrect_trees.fetch_add(1, Ordering::Relaxed);
                }
                TreeCorrectness::Mixed => {}
            }
        } else {
            self.num_unjudged_trees.fetch_add(1, Ordering::Relaxed);
        }
        let finished = self.num_finished_trees.fetch_add(1, Ordering::Relaxed) + 1;
        let num_all_correct = self.num_all_correct_trees.load(Ordering::Relaxed);
        let num_all_incorrect = self.num_all_incorrect_trees.load(Ordering::Relaxed);
        let num_unjudged = self.num_unjudged_trees.load(Ordering::Relaxed);
        let mixed = finished - num_all_correct - num_all_incorrect - num_unjudged;
        log_key_value_pair(
            "trees_correctness (✓, ❌, mixed, unjudged)",
            format!("({num_all_correct}, {num_all_incorrect}, {mixed}, {num_unjudged})"),
        );
        log_worker_progress(
            "trees",
            finished as f32 / self.total_trees_to_finish as f32,
            format!(
                "Num Trees Completed: {}/{}",
                finished, self.total_trees_to_finish
            ),
        );
    }

    fn log_database_commit(&self) {
        let committed = self.database_commit_count.fetch_add(1, Ordering::Relaxed) + 1;
        log_key_value_pair("database_commit_count", committed.to_string());
    }

    fn log_final_tree_correctness_summary(&self) {
        let num_all_correct = self.num_all_correct_trees.load(Ordering::Relaxed);
        let num_all_incorrect = self.num_all_incorrect_trees.load(Ordering::Relaxed);
        let num_unjudged = self.num_unjudged_trees.load(Ordering::Relaxed);
        let finished_trees = self.num_finished_trees.load(Ordering::Relaxed);
        let mixed = finished_trees - num_all_correct - num_all_incorrect - num_unjudged;
        log_info(format!(
            "rollout_all finished; trees_correctness (✓, ❌, mixed, unjudged) = ({num_all_correct}, {num_all_incorrect}, {mixed}, {num_unjudged})"
        ));
        log_key_value_pair(
            "trees_correctness (✓, ❌, mixed, unjudged)",
            format!("({num_all_correct}, {num_all_incorrect}, {mixed}, {num_unjudged})"),
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StopRequestedError;

struct RolloutTaskResult<S: DatasetSplit> {
    question_flat_id: QuestionFlatId<S>,
    tree_correctness: TreeCorrectness,
    has_judgments: bool,
}

async fn rollout<M: LlmModelMarker, S: DatasetSplit>(
    mut action_log: DirectTreeActionLog<M, S>,
    action_store: Arc<tokio::sync::Mutex<ActionLogStore<M, S>>>,
    llm_callable: M::Callable,
    client: Client,
    _permit: OwnedSemaphorePermit,
    start_time: Instant,
    elapsed_offset: f32,
    branching_options: BranchingRuntimeOptions,
) -> Result<RolloutTaskResult<S>, StopRequestedError> {
    let rollout_stats = RolloutStats::global();
    let _active_rollouts_guard =
        AtomicCountGuardRef::new(&rollout_stats.num_active_rollouts, "num_active_rollouts");
    let mut llm_calls_so_far = action_log
        .actions
        .iter()
        .filter(|action| action_is_llm_call(action))
        .count();
    loop {
        let tree = DirectTree::<M, S>::from_action_log(&action_log);
        if tree.completed() {
            break;
        }
        let action = tree
            .produce_action_from_direct_tree(&llm_callable, client.clone(), branching_options)
            .await?;

        match &action {
            DirectTreeAction::AttachSegmentToTreeUnjudged { final_answer, .. } => {
                let finished = rollout_stats
                    .num_finished_branches
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                let total_branches_to_finish = rollout_stats
                    .total_trees_to_finish
                    .saturating_mul(rollout_stats.trajectories_per_tree);
                log_worker_progress(
                    "branches",
                    finished as f32 / total_branches_to_finish as f32,
                    format!(
                        "Num Branches Completed: {}/{}",
                        finished, total_branches_to_finish
                    ),
                );
                match final_answer {
                    FinalAnswer::ModelProvided(_) => {
                        let _ = rollout_stats
                            .model_answers_completed
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        rollout_stats.log_model_answer_counts();
                    }
                    FinalAnswer::Failure(FailureMode::ContextWindowOverflow) => {
                        let _ = rollout_stats
                            .model_answers_context_window_overflow
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        rollout_stats.log_model_answer_counts();
                    }
                    FinalAnswer::Failure(FailureMode::OnlyEos) => {
                        let _ = rollout_stats
                            .model_answers_only_eos
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        rollout_stats.log_model_answer_counts();
                    }
                    FinalAnswer::Failure(FailureMode::TooManyTurns) => {
                        let _ = rollout_stats
                            .model_answers_too_many_turns
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        rollout_stats.log_model_answer_counts();
                    }
                }
                if let Some(trajectory_length) = trajectory_length_being_judged(&tree) {
                    rollout_stats
                        .log_trajectory_length(trajectory_length, false)
                        .await;
                }
            }
            _ => {}
        }
        if matches!(
            &action,
            DirectTreeAction::AppendSegmentContent(SegmentContent::ToolResponse(_))
        ) {
            let processed = rollout_stats
                .tool_calls_processed
                .fetch_add(1, Ordering::Relaxed)
                + 1;
            log_key_value_pair("tool_calls_processed", processed.to_string());
        }
        if action_is_llm_call(&action) {
            llm_calls_so_far += 1;
            rollout_stats
                .total_llm_calls
                .fetch_add(1, Ordering::Relaxed);
        }
        let newest_action_index = action_log.actions.len();
        action_log.actions.push(action);
        {
            let _database_waiting_guard = AtomicCountGuardRef::new(
                &rollout_stats.database_waiting_workers,
                "database_waiting_workers",
            );
            let commit_after_write = rollout_stats
                .database_commit_after_next_write
                .swap(false, Ordering::Relaxed);
            let _ = commit_after_write;
            let _did_commit = {
                let store = action_store.lock().await;
                store
                    .append(
                        action_log.question.flat_id,
                        newest_action_index,
                        action_log.actions.last().unwrap(),
                    )
                    .unwrap();
                let cumulative_elapsed =
                    elapsed_offset + (Instant::now() - start_time).as_secs_f32();
                store.write_elapsed_time(cumulative_elapsed).unwrap();
                false
            };
            if _did_commit {
                rollout_stats.log_database_commit();
            }
        }
    }
    let final_tree = DirectTree::<M, S>::from_action_log(&action_log);
    let has_judgments = !final_tree.leaf_segment_judgments.is_empty();
    let tree_correctness = final_tree.get_correctness();
    rollout_stats.log_llm_calls_per_tree(llm_calls_so_far).await;
    Ok(RolloutTaskResult {
        question_flat_id: action_log.question.flat_id,
        tree_correctness,
        has_judgments,
    })
}

pub struct RolloutProgramConfig<S: DatasetSplit> {
    pub config_nickname: String,
    pub rollout_config: RolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub client: Client,
    pub inference_endpoint: InferenceEndpoint,
    pub rollout_secs: usize,
    pub finish_all_questions: bool,
    pub total_epochs: usize,
    /// If set, open the action log store at this path instead of the default orchestrator path.
    pub action_log_store_override_path: Option<String>,
    pub use_tool: bool,
    pub fixed_temperature: NotNan<f32>,
    pub max_concurrent_rollout: usize,
    pub branching_options: BranchingRuntimeOptions,
    pub tree_artifact_output_path: Option<String>,
    pub tree_artifact_chunk_question_count: Option<usize>,
    pub question_flat_id_start: Option<usize>,
    pub question_flat_id_end: Option<usize>,
    pub question_flat_ids: Option<BTreeSet<usize>>,
}

pub struct MultiTrialRolloutProgramConfig<S: DatasetSplit> {
    pub config_nickname: String,
    pub rollout_config: RolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize,
    pub client: Client,
    pub inference_endpoint: InferenceEndpoint,
    pub rollout_secs: usize,
    pub total_epochs: usize,
    pub action_log_store_paths: Vec<String>,
    pub tree_artifact_output_paths: Vec<String>,
    pub use_tool: bool,
    pub fixed_temperature: NotNan<f32>,
    pub max_concurrent_rollout: usize,
    pub branching_options: BranchingRuntimeOptions,
    pub tree_artifact_chunk_question_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutExecutionSummary {
    pub llm_call_throughput_per_sec: f32,
    pub elapsed_secs: f32,
    pub total_llm_calls: usize,
    pub num_finished_trees: usize,
    pub num_finished_branches: usize,
    pub num_correct_branches: usize,
    pub num_all_correct_trees: usize,
    pub num_all_incorrect_trees: usize,
}

fn build_completed_tree_artifacts<M: LlmModelMarker, S: DatasetSplit>(
    store: &ActionLogStore<M, S>,
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
    rollout_config: &RolloutConfig<S>,
    posterior_calculation_config: &PosteriorCalculationConfig,
    use_tool: bool,
    fixed_temperature: NotNan<f32>,
    question_map: &BTreeMap<usize, HybridDatasetQuestion<S>>,
) -> Result<Vec<TreeArtifact<M, S>>, String> {
    let keys = store.get_keys()?;
    let mut artifacts = Vec::new();
    for key in keys {
        let Some(question) = question_map.get(&key.0).cloned() else {
            return Err(format!(
                "action log key {} does not exist in the hybrid dataset",
                key.0
            ));
        };
        let actions = store.load_action_log(key)?;
        let action_log = DirectTreeActionLog {
            mount_dir: mount_dir.to_string(),
            config_nickname: config_nickname.to_string(),
            question,
            rollout_config: rollout_config.clone(),
            posterior_calculation_config: posterior_calculation_config.clone(),
            use_tool,
            fixed_temperature,
            actions,
        };
        let tree = DirectTree::<M, S>::from_action_log(&action_log);
        if !tree.completed() {
            continue;
        }
        let artifact_id = format!(
            "{}:{}:{}:{}",
            S::dataset_file_postfix(),
            M::CLI_NAME,
            config_nickname,
            key.0
        );
        artifacts.push(TreeArtifact::from_direct_tree(
            &tree,
            artifact_id,
            M::CLI_NAME.to_string(),
            config_nickname.to_string(),
            epoch,
        ));
    }
    Ok(artifacts)
}

fn artifact_chunk_index(flat_id: usize, chunk_question_count: Option<usize>) -> usize {
    match chunk_question_count {
        Some(chunk_question_count) => {
            assert!(
                chunk_question_count > 0,
                "chunk_question_count must be positive"
            );
            flat_id / chunk_question_count
        }
        None => 0,
    }
}

fn expected_flat_ids_by_chunk_from_keys(
    question_keys: &[usize],
    chunk_question_count: Option<usize>,
) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut expected = BTreeMap::new();
    for flat_id in question_keys {
        expected
            .entry(artifact_chunk_index(*flat_id, chunk_question_count))
            .or_insert_with(BTreeSet::new)
            .insert(*flat_id);
    }
    expected
}

fn write_completed_tree_artifact_chunks<M: LlmModelMarker, S: DatasetSplit>(
    tree_artifact_output_dir: &str,
    tree_artifacts: &[TreeArtifact<M, S>],
    expected_by_chunk: &BTreeMap<usize, BTreeSet<usize>>,
    chunk_question_count: Option<usize>,
) -> Result<(), String> {
    std::fs::create_dir_all(tree_artifact_output_dir).map_err(|err| {
        format!(
            "failed to create tree artifact output dir {}: {}",
            tree_artifact_output_dir, err
        )
    })?;
    let mut artifacts_by_chunk: BTreeMap<usize, Vec<TreeArtifact<M, S>>> = BTreeMap::new();
    for artifact in tree_artifacts {
        let chunk_index = artifact_chunk_index(artifact.question.flat_id.0, chunk_question_count);
        artifacts_by_chunk
            .entry(chunk_index)
            .or_default()
            .push(artifact.clone());
    }
    for (chunk_index, artifacts) in artifacts_by_chunk {
        let chunk_path = tree_artifacts_oneshot_chunk_path(tree_artifact_output_dir, chunk_index);
        write_tree_artifacts_msgpack(&chunk_path, &artifacts)?;
        let completed_flat_ids = artifacts
            .iter()
            .map(|artifact| artifact.question.flat_id.0)
            .collect::<BTreeSet<_>>();
        let expected_flat_ids = expected_by_chunk.get(&chunk_index).ok_or_else(|| {
            format!(
                "tree artifact chunk {} has completed artifacts but no expected flat-id range",
                chunk_index
            )
        })?;
        if &completed_flat_ids == expected_flat_ids {
            let done_path =
                tree_artifacts_oneshot_chunk_done_path(tree_artifact_output_dir, chunk_index);
            std::fs::write(&done_path, b"done\n").map_err(|err| {
                format!(
                    "failed to write tree artifact done marker {}: {}",
                    done_path, err
                )
            })?;
            log_info(format!(
                "Wrote complete tree artifact chunk {} with {} trees and marker {}",
                chunk_index,
                artifacts.len(),
                done_path
            ));
        } else {
            log_warning(format!(
                "Tree artifact chunk {} is incomplete: completed {} / expected {}; marker not written",
                chunk_index,
                completed_flat_ids.len(),
                expected_flat_ids.len()
            ));
        }
    }
    Ok(())
}

fn write_completed_tree_artifact_chunk_if_ready<M: LlmModelMarker, S: DatasetSplit>(
    store: &ActionLogStore<M, S>,
    tree_artifact_output_dir: &str,
    chunk_index: usize,
    expected_by_chunk: &BTreeMap<usize, BTreeSet<usize>>,
    mount_dir: &str,
    config_nickname: &str,
    epoch: usize,
    rollout_config: &RolloutConfig<S>,
    posterior_calculation_config: &PosteriorCalculationConfig,
    use_tool: bool,
    fixed_temperature: NotNan<f32>,
    question_map: &BTreeMap<usize, HybridDatasetQuestion<S>>,
) -> Result<(), String> {
    let done_path = tree_artifacts_oneshot_chunk_done_path(tree_artifact_output_dir, chunk_index);
    if std::path::Path::new(&done_path).exists() {
        return Ok(());
    }
    let Some(expected_flat_ids) = expected_by_chunk.get(&chunk_index) else {
        return Ok(());
    };
    let mut artifacts = Vec::new();
    for flat_id in expected_flat_ids {
        let key = QuestionFlatId(*flat_id, PhantomData);
        let Ok(actions) = store.load_action_log(key) else {
            return Ok(());
        };
        let Some(question) = question_map.get(flat_id).cloned() else {
            return Err(format!(
                "missing question flat id {} in question map",
                flat_id
            ));
        };
        let action_log = DirectTreeActionLog {
            mount_dir: mount_dir.to_string(),
            config_nickname: config_nickname.to_string(),
            question,
            rollout_config: rollout_config.clone(),
            posterior_calculation_config: posterior_calculation_config.clone(),
            use_tool,
            fixed_temperature,
            actions,
        };
        let tree = DirectTree::<M, S>::from_action_log(&action_log);
        if !tree.completed() {
            return Ok(());
        }
        let artifact_id = format!(
            "{}:{}:{}:{}",
            S::dataset_file_postfix(),
            M::CLI_NAME,
            config_nickname,
            flat_id
        );
        artifacts.push(TreeArtifact::from_direct_tree(
            &tree,
            artifact_id,
            M::CLI_NAME.to_string(),
            config_nickname.to_string(),
            epoch,
        ));
    }
    let chunk_path = tree_artifacts_oneshot_chunk_path(tree_artifact_output_dir, chunk_index);
    write_tree_artifacts_msgpack(&chunk_path, &artifacts)?;
    std::fs::write(&done_path, b"done\n")
        .map_err(|err| format!("failed to write done marker {}: {}", done_path, err))?;
    log_info(format!(
        "Immediately wrote complete tree artifact chunk {} with {} trees and marker {}",
        chunk_index,
        artifacts.len(),
        done_path
    ));
    Ok(())
}

pub async fn rollout_all<M: LlmModelMarker, S: DatasetSplit>(
    mount_dir: &str,
    program_config: RolloutProgramConfig<S>,
) -> RolloutExecutionSummary {
    if let Err(error) = reset_model_answer_judgment_cache_if_any() {
        log_warning(format!(
            "Failed to reset legacy model answer judgment cache before rollout_all: {}",
            error
        ));
    }
    let RolloutProgramConfig {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        client,
        inference_endpoint,
        rollout_secs,
        finish_all_questions,
        total_epochs,
        action_log_store_override_path,
        use_tool,
        fixed_temperature,
        max_concurrent_rollout,
        branching_options,
        tree_artifact_output_path,
        tree_artifact_chunk_question_count,
        question_flat_id_start,
        question_flat_id_end,
        question_flat_ids,
    } = program_config;
    assert!(rollout_secs > 0, "rollout_secs must be positive");
    assert!(total_epochs > 0, "total_epochs must be positive");
    if finish_all_questions {
        log_info(format!(
            "rollout_all running in full-completion mode; rollout_secs={} is used only for progress reporting",
            rollout_secs
        ));
    }
    log_info(format!(
        "rollout_all using fixed_temperature={} for LLM sampling",
        fixed_temperature
    ));
    let start_time = Instant::now();

    if use_tool {
        init_python_tool_pool(4)
            .await
            .expect("failed to initialize python tool server pool");
    }
    let llm_callable = M::Callable::from_inference_endpoint(client.clone(), &inference_endpoint);
    let dataset = open_hybrid_dataset::<S>();

    let action_store = Arc::new(tokio::sync::Mutex::new(
        if let Some(ref override_path) = action_log_store_override_path {
            ActionLogStore::initialize_if_missing(override_path.clone()).unwrap_or_else(|e| {
                panic!("Failed to open action log store at {override_path}: {e}")
            })
        } else {
            open_action_logs::<M, S>(mount_dir, &config_nickname, epoch)
        },
    ));
    {
        let store = action_store.lock().await;
        store
            .write_config_bundle_if_missing(&ActionLogConfigBundle {
                rollout_config: rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
                use_tool,
                fixed_temperature,
            })
            .unwrap();
        store.sort().unwrap();
    }
    let num_questions = dataset.len();
    let question_map: BTreeMap<usize, HybridDatasetQuestion<S>> = dataset
        .iter()
        .unwrap()
        .map(|r| r.expect("failed to read question from hybrid dataset during rollout"))
        .map(|(idx, q)| (idx, q))
        .collect();
    let start_question_flat_id = question_flat_id_start.unwrap_or(0).min(num_questions);
    let end_question_flat_id = question_flat_id_end
        .unwrap_or(num_questions)
        .min(num_questions);
    assert!(
        start_question_flat_id <= end_question_flat_id,
        "question_flat_id_start ({start_question_flat_id}) must be <= question_flat_id_end ({end_question_flat_id})"
    );
    let requested_question_keys: Vec<usize> = question_flat_ids
        .map(|flat_ids| {
            flat_ids
                .into_iter()
                .filter(|flat_id| *flat_id < num_questions)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| (start_question_flat_id..end_question_flat_id).collect());
    let expected_by_chunk = expected_flat_ids_by_chunk_from_keys(
        &requested_question_keys,
        tree_artifact_chunk_question_count,
    );
    let question_keys: Vec<usize> = requested_question_keys
        .into_iter()
        .filter(|flat_id| {
            let Some(tree_artifact_output_path) = &tree_artifact_output_path else {
                return true;
            };
            let chunk_index = artifact_chunk_index(*flat_id, tree_artifact_chunk_question_count);
            let done_path =
                tree_artifacts_oneshot_chunk_done_path(tree_artifact_output_path, chunk_index);
            !std::path::Path::new(&done_path).exists()
        })
        .collect();
    log_key_value_pair("question_flat_id_start", start_question_flat_id.to_string());
    log_key_value_pair(
        "question_flat_id_end_exclusive",
        end_question_flat_id.to_string(),
    );
    if S::IS_TRAINING {
        assert!(
            epoch < total_epochs,
            "epoch ({epoch}) must be less than total_epochs ({total_epochs}) for training split"
        );
        log_key_value_pair(
            "training_segment_total_keys",
            question_keys.len().to_string(),
        );
    }
    let _rollout_all_guard = RolloutAllGuard::new(rollout_config.num_leaves, question_keys.len());
    let rollout_stats = RolloutStats::global();
    rollout_stats.reset_model_answer_judgment_cache_hit_rate();
    let (previous_elapsed, deadline, total_secs) = {
        let prev = action_store.lock().await.read_elapsed_time().unwrap_or(0.0);
        let remaining = (rollout_secs as f32 - prev).max(0.0);
        let deadline = if finish_all_questions {
            start_time + Duration::from_secs(365 * 24 * 60 * 60)
        } else {
            start_time + Duration::from_secs_f32(remaining)
        };
        if prev > 0.0 {
            log_info(format!(
                "Resuming rollout: previous elapsed={prev:.1}s, remaining={remaining:.1}s"
            ));
        }
        (prev, deadline, remaining)
    };
    let _progress_timer_handle = tokio::spawn(run_progress_timer(
        start_time,
        deadline,
        total_secs,
        mount_dir.to_string(),
        M::CLI_NAME.to_string(),
        config_nickname.clone(),
    ));

    let semaphore = Arc::new(Semaphore::new(max_concurrent_rollout));
    let mut join_set = JoinSet::new();
    let mut next_question_index = 0;
    let mut completed_flat_ids_by_chunk: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let halfway_time = start_time + (deadline - start_time) / 2;
    let mut did_set_halfway_threshold = false;

    while next_question_index < question_keys.len() || !join_set.is_empty() {
        if S::IS_TRAINING && !did_set_halfway_threshold && Instant::now() >= halfway_time {
            did_set_halfway_threshold = true;
            let num_finished_branches_so_far =
                rollout_stats.num_finished_branches.load(Ordering::Relaxed);
            let num_extra_trees_to_finish =
                num_finished_branches_so_far / rollout_stats.trajectories_per_tree;
            log_key_value_pair(
                "halfway_finished_total_branches",
                num_finished_branches_so_far.to_string(),
            );
            log_key_value_pair(
                "halfway_extra_trees_to_finish",
                num_extra_trees_to_finish.to_string(),
            );
        }

        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_question_index < question_keys.len()
                && !ROLLOUT_STOP_SIGNAL.load(Ordering::Relaxed) => {
                let permit = permit_result.expect("rollout semaphore should not be closed");

                let question_key = QuestionFlatId(question_keys[next_question_index], PhantomData);
                next_question_index += 1;
                let question = question_map
                    .get(&question_key.0)
                    .expect("question key from rollout queue must exist")
                    .clone();
                let actions = {
                    let store = action_store.lock().await;
                    store.load_or_init_action_log(question_key).unwrap()
                };

                let action_log = DirectTreeActionLog {
                    mount_dir: mount_dir.to_string(),
                    config_nickname: config_nickname.clone(),
                    question: question.clone(),
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    use_tool,
                    fixed_temperature,
                    actions,
                };
                join_set.spawn(rollout::<M, S>(
                    action_log,
                    action_store.clone(),
                    llm_callable.clone(),
                    client.clone(),
                    permit,
                    start_time,
                    previous_elapsed,
                    branching_options,
                ));
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                match joined.expect("join_set must have at least one task") {
                    Ok(Ok(task_result)) => {
                        rollout_stats.log_tree_completion(
                            task_result.tree_correctness,
                            task_result.has_judgments,
                        );
                        if let Some(tree_artifact_output_path) = &tree_artifact_output_path {
                            let chunk_index = artifact_chunk_index(
                                task_result.question_flat_id.0,
                                tree_artifact_chunk_question_count,
                            );
                            let completed_flat_ids = completed_flat_ids_by_chunk
                                .entry(chunk_index)
                                .or_default();
                            completed_flat_ids.insert(task_result.question_flat_id.0);
                            let chunk_ready = expected_by_chunk
                                .get(&chunk_index)
                                .is_some_and(|expected_flat_ids| {
                                    completed_flat_ids == expected_flat_ids
                                });
                            if chunk_ready {
                                let store = action_store.lock().await;
                                store.sort().unwrap();
                                write_completed_tree_artifact_chunk_if_ready::<M, S>(
                                    &store,
                                    tree_artifact_output_path,
                                    chunk_index,
                                    &expected_by_chunk,
                                    mount_dir,
                                    &config_nickname,
                                    epoch,
                                    &rollout_config,
                                    &posterior_calculation_config,
                                    use_tool,
                                    fixed_temperature,
                                    &question_map,
                                )
                                .unwrap_or_else(|err| {
                                    panic!("failed to write ready tree artifact chunk: {err}")
                                });
                            }
                    }
                    }
                    Ok(Err(StopRequestedError)) => {}
                    Err(join_err) => panic!("rollout task panicked: {join_err}"),
                }
            }
        }

        if ROLLOUT_STOP_SIGNAL.load(Ordering::Relaxed) && join_set.is_empty() {
            break;
        }
    }

    {
        let store = action_store.lock().await;
        store.sort().unwrap();
        if let Some(tree_artifact_output_path) = &tree_artifact_output_path {
            let tree_artifacts = build_completed_tree_artifacts::<M, S>(
                &store,
                mount_dir,
                &config_nickname,
                epoch,
                &rollout_config,
                &posterior_calculation_config,
                use_tool,
                fixed_temperature,
                &question_map,
            )
            .unwrap_or_else(|err| panic!("failed to build completed tree artifacts: {err}"));
            write_completed_tree_artifact_chunks::<M, S>(
                tree_artifact_output_path,
                &tree_artifacts,
                &expected_by_chunk,
                tree_artifact_chunk_question_count,
            )
            .unwrap_or_else(|err| panic!("failed to write tree artifact chunks: {err}"));
            log_info(format!(
                "Wrote {} completed tree artifacts under {}",
                tree_artifacts.len(),
                tree_artifact_output_path
            ));
        }
    }

    ROLLOUT_STOP_SIGNAL.store(true, Ordering::Relaxed);
    _progress_timer_handle
        .await
        .expect("progress timer task panicked");
    // restore the progress bars
    delete_worker_progress_bar("branches");
    delete_worker_progress_bar("trees");
    log_master_progress(1.0, "Rollout: time up or all finished");

    rollout_stats.log_final_tree_correctness_summary();
    rollout_stats.log_model_answer_counts();

    if let Err(error) = commit_pending_writes_if_any(mount_dir, M::CLI_NAME, &config_nickname) {
        log_warning(format!(
            "Failed to commit model answer judgment cache at the end of rollout_all: {}",
            error
        ));
    }

    let elapsed_secs = start_time.elapsed().as_secs_f32();
    let total_llm_calls = rollout_stats.total_llm_calls.load(Ordering::Relaxed);
    let llm_call_throughput_per_sec = if elapsed_secs <= f32::EPSILON {
        0.0
    } else {
        total_llm_calls as f32 / elapsed_secs
    };
    log_key_value_pair(
        "llm_call_throughput_per_sec_total",
        format!("{llm_call_throughput_per_sec:.2}"),
    );

    let num_finished_trees = rollout_stats.num_finished_trees.load(Ordering::Relaxed);
    let num_finished_branches = rollout_stats.num_finished_branches.load(Ordering::Relaxed);
    let num_correct_branches = rollout_stats.num_correct_branches.load(Ordering::Relaxed);
    let num_all_correct_trees = rollout_stats.num_all_correct_trees.load(Ordering::Relaxed);
    let num_all_incorrect_trees = rollout_stats
        .num_all_incorrect_trees
        .load(Ordering::Relaxed);

    RolloutExecutionSummary {
        llm_call_throughput_per_sec,
        elapsed_secs,
        total_llm_calls,
        num_finished_trees,
        num_finished_branches,
        num_correct_branches,
        num_all_correct_trees,
        num_all_incorrect_trees,
    }
}

pub async fn rollout_testing_trials<M: LlmModelMarker, S: DatasetSplit>(
    mount_dir: &str,
    program_config: MultiTrialRolloutProgramConfig<S>,
) -> RolloutExecutionSummary {
    if let Err(error) = reset_model_answer_judgment_cache_if_any() {
        log_warning(format!(
            "Failed to reset legacy model answer judgment cache before rollout_testing_trials: {}",
            error
        ));
    }
    let MultiTrialRolloutProgramConfig {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        client,
        inference_endpoint,
        rollout_secs,
        total_epochs,
        action_log_store_paths,
        tree_artifact_output_paths,
        use_tool,
        fixed_temperature,
        max_concurrent_rollout,
        branching_options,
        tree_artifact_chunk_question_count,
    } = program_config;
    assert!(rollout_secs > 0, "rollout_secs must be positive");
    assert!(total_epochs > 0, "total_epochs must be positive");
    assert!(
        !action_log_store_paths.is_empty(),
        "at least one testing trial is required"
    );
    assert_eq!(
        action_log_store_paths.len(),
        tree_artifact_output_paths.len(),
        "each testing trial must have one action log path and one tree artifact path"
    );
    log_info(format!(
        "rollout_testing_trials using {} trials, fixed_temperature={}, max_concurrent_rollout={}",
        action_log_store_paths.len(),
        fixed_temperature,
        max_concurrent_rollout
    ));
    let start_time = Instant::now();

    if use_tool {
        init_python_tool_pool(4)
            .await
            .expect("failed to initialize python tool server pool");
    }
    let llm_callable = M::Callable::from_inference_endpoint(client.clone(), &inference_endpoint);
    let dataset = open_hybrid_dataset::<S>();
    let num_questions = dataset.len();
    let question_map: BTreeMap<usize, HybridDatasetQuestion<S>> = dataset
        .iter()
        .unwrap()
        .map(|r| r.expect("failed to read question from hybrid dataset during testing rollout"))
        .map(|(idx, q)| (idx, q))
        .collect();
    let requested_question_keys = (0..num_questions).collect::<Vec<_>>();
    let expected_by_chunk = expected_flat_ids_by_chunk_from_keys(
        &requested_question_keys,
        tree_artifact_chunk_question_count,
    );

    let mut action_stores = Vec::new();
    for action_log_store_path in &action_log_store_paths {
        let store = ActionLogStore::<M, S>::initialize_if_missing(action_log_store_path.clone())
            .unwrap_or_else(|e| {
                panic!("Failed to open action log store at {action_log_store_path}: {e}")
            });
        store
            .write_config_bundle_if_missing(&ActionLogConfigBundle {
                rollout_config: rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
                use_tool,
                fixed_temperature,
            })
            .unwrap();
        store.sort().unwrap();
        action_stores.push(Arc::new(tokio::sync::Mutex::new(store)));
    }

    let mut work_items = Vec::new();
    for (trial_index, tree_artifact_output_path) in tree_artifact_output_paths.iter().enumerate() {
        let trial_question_keys = requested_question_keys
            .iter()
            .copied()
            .filter(|flat_id| {
                let chunk_index =
                    artifact_chunk_index(*flat_id, tree_artifact_chunk_question_count);
                let done_path =
                    tree_artifacts_oneshot_chunk_done_path(tree_artifact_output_path, chunk_index);
                !std::path::Path::new(&done_path).exists()
            })
            .collect::<Vec<_>>();
        log_key_value_pair(
            &format!("testing_trial_{trial_index}_remaining_questions"),
            trial_question_keys.len().to_string(),
        );
        for flat_id in trial_question_keys {
            work_items.push((trial_index, flat_id));
        }
    }
    log_key_value_pair(
        "testing_trial_total_remaining_questions",
        work_items.len().to_string(),
    );

    let _rollout_all_guard = RolloutAllGuard::new(rollout_config.num_leaves, work_items.len());
    let rollout_stats = RolloutStats::global();
    rollout_stats.reset_model_answer_judgment_cache_hit_rate();
    let deadline = start_time + Duration::from_secs(rollout_secs as u64);
    let _progress_timer_handle = tokio::spawn(run_progress_timer(
        start_time,
        deadline,
        rollout_secs as f32,
        mount_dir.to_string(),
        M::CLI_NAME.to_string(),
        config_nickname.clone(),
    ));

    let semaphore = Arc::new(Semaphore::new(max_concurrent_rollout));
    let mut join_set = JoinSet::new();
    let mut next_work_item_index = 0;
    let mut completed_flat_ids_by_trial_chunk: BTreeMap<(usize, usize), BTreeSet<usize>> =
        BTreeMap::new();

    while next_work_item_index < work_items.len() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_work_item_index < work_items.len()
                && !ROLLOUT_STOP_SIGNAL.load(Ordering::Relaxed) => {
                let permit = permit_result.expect("rollout semaphore should not be closed");
                let (trial_index, flat_id) = work_items[next_work_item_index];
                next_work_item_index += 1;
                let question_key = QuestionFlatId(flat_id, PhantomData);
                let question = question_map
                    .get(&flat_id)
                    .expect("question key from testing rollout queue must exist")
                    .clone();
                let action_store = action_stores[trial_index].clone();
                let actions = {
                    let store = action_store.lock().await;
                    store.load_or_init_action_log(question_key).unwrap()
                };
                let action_log = DirectTreeActionLog {
                    mount_dir: mount_dir.to_string(),
                    config_nickname: config_nickname.clone(),
                    question,
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    use_tool,
                    fixed_temperature,
                    actions,
                };
                let llm_callable = llm_callable.clone();
                let client = client.clone();
                join_set.spawn(async move {
                    rollout::<M, S>(
                        action_log,
                        action_store,
                        llm_callable,
                        client,
                        permit,
                        start_time,
                        0.0,
                        branching_options,
                    )
                    .await
                    .map(|task_result| (trial_index, task_result))
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                match joined.expect("join_set must have at least one task") {
                    Ok(Ok((trial_index, task_result))) => {
                        rollout_stats.log_tree_completion(
                            task_result.tree_correctness,
                            task_result.has_judgments,
                        );
                        let chunk_index = artifact_chunk_index(
                            task_result.question_flat_id.0,
                            tree_artifact_chunk_question_count,
                        );
                        let completed_flat_ids = completed_flat_ids_by_trial_chunk
                            .entry((trial_index, chunk_index))
                            .or_default();
                        completed_flat_ids.insert(task_result.question_flat_id.0);
                        let chunk_ready = expected_by_chunk
                            .get(&chunk_index)
                            .is_some_and(|expected_flat_ids| completed_flat_ids == expected_flat_ids);
                        if chunk_ready {
                            let store = action_stores[trial_index].lock().await;
                            store.sort().unwrap();
                            write_completed_tree_artifact_chunk_if_ready::<M, S>(
                                &store,
                                &tree_artifact_output_paths[trial_index],
                                chunk_index,
                                &expected_by_chunk,
                                mount_dir,
                                &config_nickname,
                                epoch,
                                &rollout_config,
                                &posterior_calculation_config,
                                use_tool,
                                fixed_temperature,
                                &question_map,
                            )
                            .unwrap_or_else(|err| {
                                panic!("failed to write ready testing trial tree artifact chunk: {err}")
                            });
                        }
                    }
                    Ok(Err(StopRequestedError)) => {}
                    Err(join_err) => panic!("testing rollout task panicked: {join_err}"),
                }
            }
        }

        if ROLLOUT_STOP_SIGNAL.load(Ordering::Relaxed) && join_set.is_empty() {
            break;
        }
    }

    for (trial_index, store) in action_stores.iter().enumerate() {
        let store = store.lock().await;
        store.sort().unwrap();
        let tree_artifacts = build_completed_tree_artifacts::<M, S>(
            &store,
            mount_dir,
            &config_nickname,
            epoch,
            &rollout_config,
            &posterior_calculation_config,
            use_tool,
            fixed_temperature,
            &question_map,
        )
        .unwrap_or_else(|err| panic!("failed to build completed testing tree artifacts: {err}"));
        write_completed_tree_artifact_chunks::<M, S>(
            &tree_artifact_output_paths[trial_index],
            &tree_artifacts,
            &expected_by_chunk,
            tree_artifact_chunk_question_count,
        )
        .unwrap_or_else(|err| panic!("failed to write testing trial tree artifact chunks: {err}"));
        log_info(format!(
            "Wrote {} completed testing trial {} tree artifacts under {}",
            tree_artifacts.len(),
            trial_index,
            tree_artifact_output_paths[trial_index]
        ));
    }

    ROLLOUT_STOP_SIGNAL.store(true, Ordering::Relaxed);
    _progress_timer_handle
        .await
        .expect("progress timer task panicked");
    delete_worker_progress_bar("branches");
    delete_worker_progress_bar("trees");
    log_master_progress(1.0, "Testing rollout: time up or all finished");

    rollout_stats.log_final_tree_correctness_summary();
    rollout_stats.log_model_answer_counts();

    if let Err(error) = commit_pending_writes_if_any(mount_dir, M::CLI_NAME, &config_nickname) {
        log_warning(format!(
            "Failed to commit model answer judgment cache at the end of rollout_testing_trials: {}",
            error
        ));
    }

    let elapsed_secs = start_time.elapsed().as_secs_f32();
    let total_llm_calls = rollout_stats.total_llm_calls.load(Ordering::Relaxed);
    let llm_call_throughput_per_sec = if elapsed_secs <= f32::EPSILON {
        0.0
    } else {
        total_llm_calls as f32 / elapsed_secs
    };
    log_key_value_pair(
        "llm_call_throughput_per_sec_total",
        format!("{llm_call_throughput_per_sec:.2}"),
    );

    RolloutExecutionSummary {
        llm_call_throughput_per_sec,
        elapsed_secs,
        total_llm_calls,
        num_finished_trees: rollout_stats.num_finished_trees.load(Ordering::Relaxed),
        num_finished_branches: rollout_stats.num_finished_branches.load(Ordering::Relaxed),
        num_correct_branches: rollout_stats.num_correct_branches.load(Ordering::Relaxed),
        num_all_correct_trees: rollout_stats.num_all_correct_trees.load(Ordering::Relaxed),
        num_all_incorrect_trees: rollout_stats
            .num_all_incorrect_trees
            .load(Ordering::Relaxed),
    }
}
