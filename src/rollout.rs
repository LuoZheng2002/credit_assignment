use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use arc_swap::ArcSwapOption;

use kll_rs::KllFloatSketch;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reqwest::Client;
use research_utility::progress_tui_logger::{
    delete_worker_progress_bar, log_info, log_key_value_pair, log_master_progress, log_warning,
    log_worker_progress,
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use crate::{
    atomic_count_guard::AtomicCountGuardRef,
    direct_tool::tree_action_log::{
        ActionLogConfigBundle, ActionLogStore, DirectTreeActionLog, open_action_logs,
    },
    direct_tool::{
        hybrid_dataset::{
            DatasetSplit, HybridDatasetQuestion, QuestionFlatId, open_hybrid_dataset,
        },
        posterior_calculation_config::PosteriorCalculationConfig,
        rollout_config::DirectRolloutConfig,
        trajectory::{FailureMode, FinalAnswer},
        tree::{DirectTree, SegmentContent, TreeCorrectness},
        tree_action::DirectTreeAction,
        tree_status::{
            DirectTreeStatus, GuidedBranchingSubStatus, SpontaneousBranchingSubStatus,
            TrunkSubStatus,
        },
    },
    llm_model::{InferenceEndpoint, LlmCallable, LlmModelMarker},
    model_answer_judgment_cache::commit_pending_writes_if_any,
    tool_call_python::init_python_tool_pool,
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

async fn run_progress_timer(start_time: Instant, deadline: Instant, total_secs: f32) {
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
            if let Err(error) = commit_pending_writes_if_any() {
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

    fn log_tree_completion(&self, tree_correctness: TreeCorrectness) {
        match tree_correctness {
            TreeCorrectness::AllCorrect => {
                self.num_all_correct_trees.fetch_add(1, Ordering::Relaxed);
            }
            TreeCorrectness::AllIncorrect => {
                self.num_all_incorrect_trees.fetch_add(1, Ordering::Relaxed);
            }
            TreeCorrectness::Mixed => {}
        }
        let finished = self.num_finished_trees.fetch_add(1, Ordering::Relaxed) + 1;
        let num_all_correct = self.num_all_correct_trees.load(Ordering::Relaxed);
        let num_all_incorrect = self.num_all_incorrect_trees.load(Ordering::Relaxed);
        let mixed = finished - num_all_correct - num_all_incorrect;
        log_key_value_pair(
            "trees_correctness (✓, ❌, mixed)",
            format!("({num_all_correct}, {num_all_incorrect}, {mixed})"),
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
        let finished_trees = self.num_finished_trees.load(Ordering::Relaxed);
        let mixed = finished_trees - num_all_correct - num_all_incorrect;
        log_info(format!(
            "rollout_all finished; trees_correctness (✓, ❌, mixed) = ({num_all_correct}, {num_all_incorrect}, {mixed})"
        ));
        log_key_value_pair(
            "trees_correctness (✓, ❌, mixed)",
            format!("({num_all_correct}, {num_all_incorrect}, {mixed})"),
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StopRequestedError;

async fn rollout<M: LlmModelMarker, S: DatasetSplit>(
    mut action_log: DirectTreeActionLog<M, S>,
    action_store: Arc<tokio::sync::Mutex<ActionLogStore<M, S>>>,
    llm_callable: M::Callable,
    client: Client,
    _permit: OwnedSemaphorePermit,
    start_time: Instant,
    elapsed_offset: f32,
) -> Result<TreeCorrectness, StopRequestedError> {
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
            .produce_action_from_direct_tree(&llm_callable, client.clone())
            .await?;

        match &action {
            DirectTreeAction::JudgeAnswer(correctness_judgment) => {
                let num_correct = if correctness_judgment.is_correct {
                    rollout_stats
                        .num_correct_branches
                        .fetch_add(1, Ordering::SeqCst)
                        + 1
                } else {
                    rollout_stats.num_correct_branches.load(Ordering::SeqCst)
                };
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
                let running_accuracy = num_correct as f32 / finished as f32;
                log_key_value_pair("Rollout running accuracy", running_accuracy.to_string());
                match &correctness_judgment.model_answer {
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
                        .log_trajectory_length(trajectory_length, correctness_judgment.is_correct)
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
    let tree_correctness = final_tree.get_correctness();
    rollout_stats.log_llm_calls_per_tree(llm_calls_so_far).await;
    Ok(tree_correctness)
}

pub struct RolloutProgramConfig<S: DatasetSplit> {
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub client: Client,
    pub max_rollout_concurrency: usize,
    pub inference_endpoint: InferenceEndpoint,
    pub rollout_time_limit_secs: usize,
    pub max_python_processes: usize,
    pub total_epochs: usize,
    /// If set, open the action log store at this path instead of the default orchestrator path.
    pub action_log_store_override_path: Option<String>,
}

pub struct RolloutExecutionSummary {
    pub llm_call_throughput_per_sec: f32,
    pub elapsed_secs: f32,
    pub total_llm_calls: usize,
}

pub async fn rollout_all<M: LlmModelMarker, S: DatasetSplit>(
    program_config: RolloutProgramConfig<S>,
) -> RolloutExecutionSummary {
    let RolloutProgramConfig {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        client,
        max_rollout_concurrency,
        inference_endpoint,
        rollout_time_limit_secs,
        max_python_processes,
        total_epochs,
        action_log_store_override_path,
    } = program_config;
    assert!(
        max_python_processes > 0,
        "max_python_processes must be positive"
    );
    assert!(
        rollout_time_limit_secs > 0,
        "rollout_time_limit_secs must be positive"
    );
    assert!(
        max_rollout_concurrency > 0,
        "max_rollout_concurrency must be positive"
    );
    assert!(total_epochs > 0, "total_epochs must be positive");
    log_info(format!(
        "rollout_all using fixed_temperature={} for LLM sampling",
        rollout_config.fixed_temperature
    ));
    let start_time = Instant::now();

    if rollout_config.use_tool {
        init_python_tool_pool(max_python_processes)
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
            open_action_logs::<M, S>(&config_nickname, epoch)
        },
    ));
    {
        let store = action_store.lock().await;
        store
            .write_config_bundle_if_missing(&ActionLogConfigBundle {
                rollout_config: rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
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
    let mut question_keys: Vec<usize> = (0..num_questions).collect();
    if S::IS_TRAINING {
        assert!(
            epoch < total_epochs,
            "epoch ({epoch}) must be less than total_epochs ({total_epochs}) for training split"
        );
        let training_segment_start = if question_keys.is_empty() {
            0
        } else {
            let mut training_segment_rng = StdRng::seed_from_u64(epoch as u64);
            training_segment_rng.random_range(0..question_keys.len())
        };
        question_keys.rotate_left(training_segment_start);
        log_key_value_pair(
            "training_segment_start_index",
            training_segment_start.to_string(),
        );
        log_key_value_pair("training_segment_start_seed", epoch.to_string());
        log_key_value_pair(
            "training_segment_total_keys",
            question_keys.len().to_string(),
        );
    }
    let _rollout_all_guard = RolloutAllGuard::new(
        rollout_config.max_num_total_trajectories,
        question_keys.len(),
    );
    let rollout_stats = RolloutStats::global();
    rollout_stats.reset_model_answer_judgment_cache_hit_rate();
    let (previous_elapsed, deadline, total_secs) = {
        let prev = action_store.lock().await.read_elapsed_time().unwrap_or(0.0);
        let remaining = (rollout_time_limit_secs as f32 - prev).max(0.0);
        let deadline = start_time + Duration::from_secs_f32(remaining);
        if prev > 0.0 {
            log_info(format!(
                "Resuming rollout: previous elapsed={prev:.1}s, remaining={remaining:.1}s"
            ));
        }
        (prev, deadline, remaining)
    };
    let _progress_timer_handle = tokio::spawn(run_progress_timer(start_time, deadline, total_secs));

    let semaphore = Arc::new(Semaphore::new(max_rollout_concurrency));
    let mut join_set = JoinSet::new();
    let mut next_question_index = 0;
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
                    question: question.clone(),
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
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
                ));
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                match joined.expect("join_set must have at least one task") {
                    Ok(Ok(tree_correctness)) => {
                        rollout_stats.log_tree_completion(tree_correctness);
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

    if let Err(error) = commit_pending_writes_if_any() {
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

    RolloutExecutionSummary {
        llm_call_throughput_per_sec,
        elapsed_secs,
        total_llm_calls,
    }
}
