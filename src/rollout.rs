use std::collections::VecDeque;

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use kll_rs::KllFloatSketch;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reqwest::Client;
use research_utility::progress_tui_logger::{
    delete_worker_progress_bar, log_info, log_key_value_pair, log_master_progress,
    log_worker_progress,
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use crate::atomic_count_guard::AtomicCountGuard;
use crate::direct_tool::tree_action_log::DirectTreeActionLog;
use crate::{
    direct_tool::{
        hybrid_dataset::{DatasetSplit, open_hybrid_dataset},
        posterior_calculation_config::PosteriorCalculationConfig,
        rollout_config::DirectRolloutConfig,
        tree::{DirectTree, SegmentContent},
        tree_action::DirectTreeAction,
        tree_action_log::{ActionStoreAdapter, open_action_logs},
        tree_status::{
            DirectTreeStatus, GuidedBranchingSubStatus, SpontaneousBranchingSubStatus,
            TrunkSubStatus,
        },
    },
    llm_model::{LlmCallable, LlmCliArgs, LlmModelMarker},
    tool_call_python::PythonToolServerPool,
};

struct DistributionStats {
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

fn all_same_correctness(judgments: impl IntoIterator<Item = bool>) -> Option<bool> {
    let mut judgments = judgments.into_iter();
    let first = judgments.next()?;
    if judgments.all(|is_correct| is_correct == first) {
        Some(first)
    } else {
        None
    }
}

fn classify_all_same_trajectory_tree<M: LlmModelMarker, S: DatasetSplit>(
    tree: &DirectTree<M, S>,
) -> Option<bool> {
    if !S::IS_TRAINING {
        return None;
    }
    all_same_correctness(
        tree.leaf_segment_judgments
            .values()
            .map(|judgment| judgment.is_correct),
    )
}

async fn run_progress_timer(
    start_time: Instant,
    deadline: Instant,
    total_secs: f32,
    stop_signal: Arc<AtomicBool>,
    total_llm_calls: Arc<AtomicUsize>,
) {
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
    let mut llm_call_samples: VecDeque<(Instant, usize)> = VecDeque::new();
    loop {
        if stop_signal.load(Ordering::Relaxed) {
            break;
        }
        let now = Instant::now();
        if now >= deadline {
            stop_signal.store(true, Ordering::Relaxed);
            break;
        }
        if now >= next_progress_log_time {
            log_time_progress(now);
            // If the runtime was blocked for a while, avoid "catch-up" bursts by
            // scheduling from the current time rather than replaying missed ticks.
            next_progress_log_time = now + progress_log_interval;
        }

        if now >= next_throughput_log_time {
            let current_llm_calls = total_llm_calls.load(Ordering::Relaxed);
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

        let wake_time = std::cmp::min(
            std::cmp::min(
                std::cmp::min(next_progress_log_time, next_throughput_log_time),
                deadline,
            ),
            Instant::now() + Duration::from_millis(100),
        );
        sleep_until(wake_time).await;
    }

    log_time_progress(Instant::now());
}

#[derive(Clone)]
pub struct RolloutSharedStates {
    python_tool_pool: Arc<PythonToolServerPool>,
    sglang_waiting_workers: Arc<AtomicUsize>,
    judge_waiting_workers: Arc<AtomicUsize>,
    tool_waiting_workers: Arc<AtomicUsize>,
    sqlite_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
    num_finished_branches: Arc<AtomicUsize>,
    num_finished_trees: Arc<AtomicUsize>,
    num_correct_branches: Arc<AtomicUsize>,
    num_all_correct_trees: Arc<AtomicUsize>,
    num_all_incorrect_trees: Arc<AtomicUsize>,
    llm_call_stats: Arc<parking_lot::RwLock<DistributionStats>>,
    trajectory_length_stats: Arc<parking_lot::RwLock<DistributionStats>>,
    correct_trajectory_length_stats: Arc<parking_lot::RwLock<DistributionStats>>,
    num_active_rollouts: Arc<AtomicUsize>,
    total_llm_calls: Arc<AtomicUsize>,
    tool_calls_processed: Arc<AtomicUsize>,
    trajectories_per_tree: usize,
    total_trees_to_finish: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StopRequestedError;

async fn rollout<M: LlmModelMarker, S: DatasetSplit>(
    mut action_log: DirectTreeActionLog<M, S>,
    action_store: ActionStoreAdapter<M, S>,
    llm_callable: M::Callable,
    client: Client,
    shared_states: RolloutSharedStates,
    _active_rollouts_guard: AtomicCountGuard,
    _permit: OwnedSemaphorePermit,
) -> Result<(), StopRequestedError> {
    let RolloutSharedStates {
        python_tool_pool,
        sglang_waiting_workers,
        judge_waiting_workers,
        tool_waiting_workers,
        sqlite_waiting_workers,
        stop_signal,
        num_finished_branches,
        num_finished_trees,
        num_correct_branches,
        num_all_correct_trees,
        num_all_incorrect_trees,
        llm_call_stats,
        trajectory_length_stats,
        correct_trajectory_length_stats,
        num_active_rollouts,
        total_llm_calls,
        tool_calls_processed,
        trajectories_per_tree,
        total_trees_to_finish,
    } = shared_states;

    let _ = num_active_rollouts;
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
            .produce_action_from_direct_tree(
                &llm_callable,
                client.clone(),
                python_tool_pool.clone(),
                sglang_waiting_workers.clone(),
                judge_waiting_workers.clone(),
                tool_waiting_workers.clone(),
                stop_signal.clone(),
            )
            .await?;
        // to do: put it to the to_action function
        match &action {
            DirectTreeAction::JudgeAnswer(correctness_judgment) => {
                let num_correct = if correctness_judgment.is_correct {
                    num_correct_branches.fetch_add(1, Ordering::SeqCst) + 1
                } else {
                    num_correct_branches.load(Ordering::SeqCst)
                };
                let finished =
                    num_finished_branches.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let total_branches_to_finish =
                    total_trees_to_finish.saturating_mul(trajectories_per_tree);
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
                if let Some(trajectory_length) = trajectory_length_being_judged(&tree) {
                    let summary = trajectory_length_stats
                        .write()
                        .update_and_get_summary(trajectory_length);
                    log_key_value_pair(
                        "trajectory_length (min, median, q3, max)",
                        condensed_distribution(&summary),
                    );
                    if correctness_judgment.is_correct {
                        let correct_summary = correct_trajectory_length_stats
                            .write()
                            .update_and_get_summary(trajectory_length);
                        log_key_value_pair(
                            "correct_trajectory_length (min, median, q3, max)",
                            condensed_distribution(&correct_summary),
                        );
                    }
                }
            }
            _ => {}
        }
        if matches!(
            &action,
            DirectTreeAction::AppendSegmentContent(SegmentContent::ToolResponse(_))
        ) {
            let processed = tool_calls_processed.fetch_add(1, Ordering::Relaxed) + 1;
            log_key_value_pair("tool_calls_processed", processed.to_string());
        }
        if action_is_llm_call(&action) {
            llm_calls_so_far += 1;
            total_llm_calls.fetch_add(1, Ordering::Relaxed);
        }
        let newest_action_index = action_log.actions.len();
        action_log.actions.push(action);
        {
            // add sqlite waiting worker guard
            let _sqlite_waiting_guard =
                AtomicCountGuard::new(sqlite_waiting_workers.clone(), "sqlite_waiting_workers");
            action_store
                .append_action_at(
                    action_log.question.flat_id,
                    newest_action_index,
                    action_log.actions.last().unwrap(),
                )
                .await
                .unwrap();
        }
    }
    let final_tree = DirectTree::<M, S>::from_action_log(&action_log);
    if let Some(all_correct) = classify_all_same_trajectory_tree(&final_tree) {
        if all_correct {
            num_all_correct_trees.fetch_add(1, Ordering::Relaxed);
        } else {
            num_all_incorrect_trees.fetch_add(1, Ordering::Relaxed);
        }
    }
    let finished = num_finished_trees.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let num_all_correct = num_all_correct_trees.load(Ordering::Relaxed);
    let num_all_incorrect = num_all_incorrect_trees.load(Ordering::Relaxed);
    let mixed = finished - num_all_correct - num_all_incorrect;
    log_key_value_pair(
        "trees_correctness (✓, ❌, mixed)",
        format!("({num_all_correct}, {num_all_incorrect}, {mixed})"),
    );
    log_worker_progress(
        "trees",
        finished as f32 / total_trees_to_finish as f32,
        format!(
            "Num Trees Completed: {}/{}",
            finished, total_trees_to_finish
        ),
    );
    let summary = llm_call_stats
        .write()
        .update_and_get_summary(llm_calls_so_far);
    log_key_value_pair(
        "llm_calls_per_tree (min, median, q3, max)",
        condensed_distribution(&summary),
    );
    Ok(())
}

pub struct RolloutProgramConfig<S: DatasetSplit> {
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig<S>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub client: Client,
    pub max_rollout_concurrency: usize,
    pub llm_cli_args: LlmCliArgs,
    pub rollout_time_limit_secs: usize,
    pub max_python_processes: usize,
    pub total_epochs: usize,
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
        llm_cli_args,
        rollout_time_limit_secs,
        max_python_processes,
        total_epochs,
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
    let rollout_time_limit = Duration::from_secs(rollout_time_limit_secs as u64);
    let start_time = Instant::now();
    let deadline = start_time + rollout_time_limit;
    let total_secs = rollout_time_limit_secs as f32;

    let python_tool_pool = Arc::new(
        PythonToolServerPool::new(max_python_processes)
            .await
            .expect("failed to initialize python tool server pool"),
    );
    let llm_callable = M::Callable::from_cli_args(client.clone(), &llm_cli_args);
    let dataset = open_hybrid_dataset::<S>();

    // let DirectTreeActionLogStore {
    //     metadata_store,
    //     action_store,
    //     _phantom,
    // } = DirectTreeActionLogStore::<M>::initialize_if_missing(
    //     action_logs_file_path::<M, S>(&config_nickname, epoch),
    // );
    let action_store = open_action_logs::<M, S>(&config_nickname, epoch);
    let action_store_adapter = ActionStoreAdapter::new(action_store);
    let mut question_keys = dataset.get_keys().unwrap();
    // sort by question id to ensure deterministic order
    question_keys.sort();
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
    let stop_signal = Arc::new(AtomicBool::new(false));
    let sglang_waiting_workers = Arc::new(AtomicUsize::new(0));
    let judge_waiting_workers = Arc::new(AtomicUsize::new(0));
    let tool_waiting_workers = Arc::new(AtomicUsize::new(0));
    let sqlite_waiting_workers = Arc::new(AtomicUsize::new(0));
    let num_finished_branches = Arc::new(AtomicUsize::new(0));
    let num_finished_trees = Arc::new(AtomicUsize::new(0));
    let num_correct_branches = Arc::new(AtomicUsize::new(0));
    let num_all_correct_trees = Arc::new(AtomicUsize::new(0));
    let num_all_incorrect_trees = Arc::new(AtomicUsize::new(0));
    let num_active_rollouts = Arc::new(AtomicUsize::new(0));
    let llm_call_stats = Arc::new(parking_lot::RwLock::new(DistributionStats::new()));
    let trajectory_length_stats = Arc::new(parking_lot::RwLock::new(DistributionStats::new()));
    let correct_trajectory_length_stats =
        Arc::new(parking_lot::RwLock::new(DistributionStats::new()));
    let total_llm_calls = Arc::new(AtomicUsize::new(0));
    let tool_calls_processed = Arc::new(AtomicUsize::new(0));
    let total_trees_to_finish = question_keys.len();

    let shared_states = RolloutSharedStates {
        python_tool_pool,
        sglang_waiting_workers,
        judge_waiting_workers,
        tool_waiting_workers,
        sqlite_waiting_workers,
        stop_signal,
        num_finished_branches,
        num_finished_trees,
        num_correct_branches,
        num_all_correct_trees,
        num_all_incorrect_trees,
        llm_call_stats,
        trajectory_length_stats,
        correct_trajectory_length_stats,
        num_active_rollouts,
        total_llm_calls,
        tool_calls_processed,
        trajectories_per_tree: rollout_config.max_num_total_trajectories,
        total_trees_to_finish,
    };

    let progress_timer_handle = tokio::spawn(run_progress_timer(
        start_time,
        deadline,
        total_secs,
        shared_states.stop_signal.clone(),
        shared_states.total_llm_calls.clone(),
    ));

    let semaphore = Arc::new(Semaphore::new(max_rollout_concurrency));
    let mut join_set = JoinSet::new();
    let mut next_question_index = 0;
    let halfway_time = start_time + rollout_time_limit / 2;
    let mut did_set_halfway_threshold = false;
    let mut halfway_question_queue_size = question_keys.len();
    let extra_question_active_rollout_threshold = max_rollout_concurrency.min(20);
    let mut entered_extra_question_phase = false;

    while next_question_index < question_keys.len() || !join_set.is_empty() {
        if S::IS_TRAINING && !did_set_halfway_threshold && Instant::now() >= halfway_time {
            did_set_halfway_threshold = true;
            let num_finished_branches_so_far =
                shared_states.num_finished_branches.load(Ordering::Relaxed);
            let num_extra_trees_to_finish =
                num_finished_branches_so_far / shared_states.trajectories_per_tree;
            halfway_question_queue_size = next_question_index
                .saturating_add(num_extra_trees_to_finish)
                .min(question_keys.len());
            log_key_value_pair(
                "halfway_finished_total_branches",
                num_finished_branches_so_far.to_string(),
            );
            log_key_value_pair(
                "halfway_extra_trees_to_finish",
                num_extra_trees_to_finish.to_string(),
            );
            log_key_value_pair(
                "halfway_question_queue_size",
                halfway_question_queue_size.to_string(),
            );
        }

        let finished_trees = shared_states.num_finished_trees.load(Ordering::Relaxed);
        let in_extra_question_phase = S::IS_TRAINING
            && did_set_halfway_threshold
            && finished_trees >= halfway_question_queue_size;
        if in_extra_question_phase && !entered_extra_question_phase {
            entered_extra_question_phase = true;
            log_key_value_pair(
                "extra_question_active_rollout_threshold",
                extra_question_active_rollout_threshold.to_string(),
            );
        }
        let current_question_queue_limit = if in_extra_question_phase {
            question_keys.len()
        } else {
            halfway_question_queue_size
        };
        let current_active_rollout_threshold = if in_extra_question_phase {
            extra_question_active_rollout_threshold
        } else {
            max_rollout_concurrency
        };

        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_question_index < current_question_queue_limit
                && !shared_states.stop_signal.load(Ordering::Relaxed)
                && shared_states.num_active_rollouts.load(Ordering::Relaxed) < current_active_rollout_threshold => {
                let permit = permit_result.expect("rollout semaphore should not be closed");
                let Some(active_rollouts_guard) = AtomicCountGuard::try_new_with_max(
                    shared_states.num_active_rollouts.clone(),
                    "num_active_rollouts",
                    current_active_rollout_threshold,
                ) else {
                    continue;
                };
                let question_key = question_keys[next_question_index];
                next_question_index += 1;
                let question = dataset
                    .get(question_key)
                    .unwrap()
                    .expect("question key from rollout queue must exist");
                let actions = action_store_adapter.get_or_init_actions(question_key).await.unwrap();

                let action_log = DirectTreeActionLog {
                    question: question.clone(),
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                    actions,
                };
                join_set.spawn(rollout::<M, S>(
                    action_log,
                    action_store_adapter.clone(),
                    llm_callable.clone(),
                    client.clone(),
                    shared_states.clone(),
                    active_rollouts_guard,
                    permit,
                ));
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                match joined.expect("join_set must have at least one task") {
                    Ok(Ok(())) | Ok(Err(StopRequestedError)) => {}
                    Err(join_err) => panic!("rollout task panicked: {join_err}"),
                }
            }
        }

        if shared_states.stop_signal.load(Ordering::Relaxed) && join_set.is_empty() {
            break;
        }
    }

    shared_states.stop_signal.store(true, Ordering::Relaxed);
    progress_timer_handle
        .await
        .expect("progress timer task panicked");
    // restore the progress bars
    delete_worker_progress_bar("branches");
    delete_worker_progress_bar("trees");
    log_master_progress(1.0, "Rollout: time up or all finished");

    let num_all_correct = shared_states.num_all_correct_trees.load(Ordering::Relaxed);
    let num_all_incorrect = shared_states
        .num_all_incorrect_trees
        .load(Ordering::Relaxed);
    let finished_trees = shared_states.num_finished_trees.load(Ordering::Relaxed);
    let mixed = finished_trees - num_all_correct - num_all_incorrect;
    log_info(format!(
        "Rollout_all finished; trees_correctness (✓, ❌, mixed) = ({num_all_correct}, {num_all_incorrect}, {mixed})"
    ));
    log_key_value_pair(
        "trees_correctness (✓, ❌, mixed)",
        format!("({num_all_correct}, {num_all_incorrect}, {mixed})"),
    );

    let elapsed_secs = start_time.elapsed().as_secs_f32();
    let total_llm_calls = shared_states.total_llm_calls.load(Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::all_same_correctness;

    #[test]
    fn all_same_correctness_handles_empty_and_single_leaf_trees() {
        assert_eq!(all_same_correctness([]), None);
        assert_eq!(all_same_correctness([true]), Some(true));
        assert_eq!(all_same_correctness([false]), Some(false));
    }

    #[test]
    fn all_same_correctness_rejects_mixed_values() {
        assert_eq!(all_same_correctness([true, true, true]), Some(true));
        assert_eq!(all_same_correctness([false, false]), Some(false));
        assert_eq!(all_same_correctness([true, false]), None);
        assert_eq!(all_same_correctness([false, true]), None);
    }
}
