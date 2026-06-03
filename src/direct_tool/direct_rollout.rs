use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use kll_rs::KllFloatSketch;
use reqwest::Client;
use research_utility::{
    asset_file::AssetFile,
    progress_tui_server::{
        delete_worker_progress_bar, log_key_value_pair, log_master_progress, log_worker_progress,
    },
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use crate::atomic_count_guard::AtomicCountGuard;
use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::{DirectTree, SegmentContent},
        direct_tree_action::DirectTreeAction,
        direct_tree_action_log::{
            AssetFileDirectTreeActionLogs, DirectTreeActionLogMetadata, DirectTreeActionLogStore,
        },
        hybrid_dataset::{AssetFileHybridDataset, HybridDatasetQuestion, HybridDatasetStore},
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    llm_model::{LlmCallable, LlmCliArgs, LlmModelMarker},
    tool_call_python::PythonToolServerPool,
};

struct LlmCallStats {
    sketch: KllFloatSketch,
    num_samples: usize,
    min: usize,
    max: usize,
}

impl LlmCallStats {
    fn new() -> Self {
        Self {
            sketch: KllFloatSketch::new().expect("failed to initialize KLL sketch"),
            num_samples: 0,
            min: 0,
            max: 0,
        }
    }

    fn update_and_get_summary(&mut self, llm_calls: usize) -> LlmCallSummary {
        self.sketch.update(llm_calls as f32);
        if self.num_samples == 0 {
            self.min = llm_calls;
            self.max = llm_calls;
        } else {
            self.min = std::cmp::min(self.min, llm_calls);
            self.max = std::cmp::max(self.max, llm_calls);
        }
        self.num_samples += 1;

        LlmCallSummary {
            min: self.min,
            median: quantile_to_usize(self.sketch.get_quantile(0.5)),
            q3: quantile_to_usize(self.sketch.get_quantile(0.75)),
            max: self.max,
        }
    }
}

struct LlmCallSummary {
    min: usize,
    median: usize,
    q3: usize,
    max: usize,
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
            next_progress_log_time += Duration::from_secs(1);
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
            next_throughput_log_time += throughput_log_interval;
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
    llm_call_stats: Arc<parking_lot::RwLock<LlmCallStats>>,
    num_active_rollouts: Arc<AtomicUsize>,
    total_llm_calls: Arc<AtomicUsize>,
    total_branches_to_finish: usize,
    total_trees_to_finish: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StopRequestedError;

async fn rollout<M: LlmModelMarker>(
    question: HybridDatasetQuestion,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    rollout_store: DirectTreeActionLogStore<M>,
    llm_callable: M::Callable,
    client: Client,
    shared_states: RolloutSharedStates,
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
        llm_call_stats,
        num_active_rollouts,
        total_llm_calls,
        total_branches_to_finish,
        total_trees_to_finish,
    } = shared_states;

    let _active_rollouts_guard =
        AtomicCountGuard::new(num_active_rollouts.clone(), "num_active_rollouts");
    {
        let _sqlite_waiting_workers_guard =
            AtomicCountGuard::new(sqlite_waiting_workers.clone(), "sqlite_waiting_workers");
        rollout_store
            .get_or_init_metadata(
                question.flat_id,
                &DirectTreeActionLogMetadata {
                    question: question.clone(),
                    rollout_config: rollout_config.clone(),
                    posterior_calculation_config: posterior_calculation_config.clone(),
                },
            )
            .await
            .unwrap();
    }
    let mut action_log = {
        let _sqlite_waiting_workers_guard =
            AtomicCountGuard::new(sqlite_waiting_workers.clone(), "sqlite_waiting_workers");
        rollout_store
            .get(question.flat_id)
            .await
            .unwrap()
            .expect("metadata must exist right after initialization")
    };
    let mut llm_calls_so_far = action_log
        .actions
        .iter()
        .filter(|action| action_is_llm_call(action))
        .count();
    loop {
        let tree = DirectTree::<M>::from_action_log(&action_log);
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
            }
            _ => {}
        }
        if action_is_llm_call(&action) {
            llm_calls_so_far += 1;
            total_llm_calls.fetch_add(1, Ordering::Relaxed);
        }
        let newest_action_index = action_log.actions.len();
        action_log.actions.push(action);
        {
            let _sqlite_waiting_workers_guard =
                AtomicCountGuard::new(sqlite_waiting_workers.clone(), "sqlite_waiting_workers");
            rollout_store
                .append_action_at(
                    question.flat_id,
                    newest_action_index,
                    action_log.actions.last().unwrap(),
                )
                .await
                .unwrap();
        }
    }
    // log_info(format!("Rollout {} finished", question.flat_id));
    let finished = num_finished_trees.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
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
    log_key_value_pair("min_llm_calls_per_tree", summary.min.to_string());
    log_key_value_pair("median_llm_calls_per_tree", summary.median.to_string());
    log_key_value_pair("third_quartile_llm_calls_per_tree", summary.q3.to_string());
    log_key_value_pair("max_llm_calls_per_tree", summary.max.to_string());
    Ok(())
}

async fn rollout_task<M: LlmModelMarker>(
    question_key: usize,
    dataset: Arc<HybridDatasetStore>,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    rollout_store: DirectTreeActionLogStore<M>,
    llm_callable: M::Callable,
    client: Client,
    shared_states: RolloutSharedStates,
    _permit: OwnedSemaphorePermit,
) -> Result<(), StopRequestedError> {
    let question = dataset
        .get(question_key)
        .await
        .unwrap()
        .expect("question key from rollout queue must exist");
    rollout::<M>(
        question,
        rollout_config,
        posterior_calculation_config,
        rollout_store,
        llm_callable,
        client,
        shared_states,
    )
    .await
}

pub struct RolloutProgramConfig {
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub client: Client,
    pub max_rollout_concurrency: usize,
    pub llm_cli_args: LlmCliArgs,
    pub rollout_time_limit_secs: usize,
    pub num_python_tool_servers: usize,
}

pub async fn rollout_all<M: LlmModelMarker>(program_config: RolloutProgramConfig) {
    let RolloutProgramConfig {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        client,
        max_rollout_concurrency,
        llm_cli_args,
        rollout_time_limit_secs,
        num_python_tool_servers,
    } = program_config;
    assert!(
        num_python_tool_servers > 0,
        "num_python_tool_servers must be positive"
    );
    assert!(
        rollout_time_limit_secs > 0,
        "rollout_time_limit_secs must be positive"
    );
    assert!(
        max_rollout_concurrency > 0,
        "max_rollout_concurrency must be positive"
    );
    let rollout_time_limit = Duration::from_secs(rollout_time_limit_secs as u64);
    let start_time = Instant::now();
    let deadline = start_time + rollout_time_limit;
    let total_secs = rollout_time_limit_secs as f32;

    let python_tool_pool = Arc::new(
        PythonToolServerPool::new(num_python_tool_servers)
            .await
            .expect("failed to initialize python tool server pool"),
    );
    let llm_callable = M::Callable::from_cli_args(client.clone(), &llm_cli_args);
    let asset_file_dataset = AssetFileHybridDataset {
        split: rollout_config.split.clone(),
    };
    let dataset = Arc::new(asset_file_dataset.fetch().await);
    let asset_file_action_logs = AssetFileDirectTreeActionLogs::<M> {
        nickname: config_nickname,
        rollout_config: rollout_config.clone(),
        posterior_calculation_config: posterior_calculation_config.clone(),
        epoch,
        _phantom: std::marker::PhantomData,
    };
    asset_file_action_logs.delete_target_file_if_stale();
    asset_file_action_logs.create_tracking_file();
    let rollout_store =
        DirectTreeActionLogStore::<M>::initialize_if_missing(asset_file_action_logs.file_path())
            .await;
    let mut question_keys = dataset.get_keys().await.unwrap();
    // sort by question id to ensure deterministic order
    question_keys.sort();
    let stop_signal = Arc::new(AtomicBool::new(false));
    let sglang_waiting_workers = Arc::new(AtomicUsize::new(0));
    let judge_waiting_workers = Arc::new(AtomicUsize::new(0));
    let tool_waiting_workers = Arc::new(AtomicUsize::new(0));
    let sqlite_waiting_workers = Arc::new(AtomicUsize::new(0));
    let num_finished_branches = Arc::new(AtomicUsize::new(0));
    let num_finished_trees = Arc::new(AtomicUsize::new(0));
    let num_correct_branches = Arc::new(AtomicUsize::new(0));
    let num_active_rollouts = Arc::new(AtomicUsize::new(0));
    let llm_call_stats = Arc::new(parking_lot::RwLock::new(LlmCallStats::new()));
    let total_llm_calls = Arc::new(AtomicUsize::new(0));
    let total_branches_to_finish = question_keys.len() * rollout_config.max_num_total_trajectories;
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
        llm_call_stats,
        num_active_rollouts,
        total_llm_calls,
        total_branches_to_finish,
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

    while next_question_index < question_keys.len() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_question_index < question_keys.len() && !shared_states.stop_signal.load(Ordering::Relaxed) => {
                let permit = permit_result.expect("rollout semaphore should not be closed");
                let question_key = question_keys[next_question_index];
                next_question_index += 1;
                join_set.spawn(rollout_task::<M>(
                    question_key,
                    dataset.clone(),
                    rollout_config.clone(),
                    posterior_calculation_config.clone(),
                    rollout_store.clone(),
                    llm_callable.clone(),
                    client.clone(),
                    shared_states.clone(),
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
}
