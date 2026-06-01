use std::sync::{
    Arc,
    RwLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::collections::VecDeque;

use kll_rs::KllFloatSketch;
use reqwest::Client;
use research_utility::{
    asset_file::AssetFile,
    log_message::{
        delete_worker_progress_bar, log_key_value_pair, log_master_progress, log_worker_progress,
    },
    sqlite_store::{SqliteBusyRetryConfig, SqliteStore},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep, sleep_until};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::{DirectTree, SegmentContent},
        direct_tree_action::DirectTreeAction,
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        hybrid_dataset::{AssetFileHybridDataset, HybridDatasetQuestion, HybridDatasetStore},
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    llm_model::{LlmCallable, LlmCliArgs, LlmModelMarker},
    tool_call_python::PythonToolServerPool,
};

struct RolloutExecutionContext<M: LlmModelMarker> {
    question_keys: Vec<usize>,
    dataset: HybridDatasetStore,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    rollout_store: SqliteStore<usize, DirectTreeActionLog<M>>,
    llm_callable: M::Callable,
    client: Client,
    question_semaphore: Arc<Semaphore>,
    python_tool_pool: Arc<PythonToolServerPool>,
    sglang_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
    num_finished_branches: Arc<AtomicUsize>,
    num_finished_trees: Arc<AtomicUsize>,
    num_correct_branches: Arc<AtomicUsize>,
    llm_call_stats: Arc<RwLock<LlmCallStats>>,
    num_requeued: Arc<AtomicUsize>,
    num_long_running_rollouts: Arc<AtomicUsize>,
    num_active_rollouts: Arc<AtomicUsize>,
    max_rollout_concurrency: usize,
    max_num_long_running_rollouts: usize,
    total_branches_to_finish: usize,
    total_trees_to_finish: usize,
}

enum RolloutOutcome {
    Completed,
    Requeue { flat_id: usize },
}

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

    fn q3(&self) -> Option<usize> {
        if self.num_samples == 0 {
            return None;
        }
        Some(quantile_to_usize(self.sketch.get_quantile(0.75)))
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

struct LongRunningRolloutGuard {
    num_long_running_rollouts: Arc<AtomicUsize>,
    has_slot: bool,
}

impl LongRunningRolloutGuard {
    fn new(num_long_running_rollouts: Arc<AtomicUsize>) -> Self {
        Self {
            num_long_running_rollouts,
            has_slot: false,
        }
    }

    fn has_slot(&self) -> bool {
        self.has_slot
    }

    fn try_acquire_slot(&mut self, max_num_long_running_rollouts: usize) -> bool {
        if self.has_slot {
            return true;
        }

        let mut current = self.num_long_running_rollouts.load(Ordering::SeqCst);
        loop {
            if current >= max_num_long_running_rollouts {
                return false;
            }
            match self.num_long_running_rollouts.compare_exchange(
                current,
                current + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    self.has_slot = true;
                    let new_count = current + 1;
                    log_key_value_pair("num_long_running_questions", new_count.to_string());
                    return true;
                }
                Err(actual) => {
                    current = actual;
                }
            }
        }
    }
}

impl Drop for LongRunningRolloutGuard {
    fn drop(&mut self) {
        if self.has_slot {
            let previous = self.num_long_running_rollouts.fetch_sub(1, Ordering::SeqCst);
            assert!(previous > 0, "num_long_running_rollouts underflow");
            let new_count = previous - 1;
            log_key_value_pair("num_long_running_questions", new_count.to_string());
        }
    }
}

struct ActiveRolloutGuard {
    num_active_rollouts: Arc<AtomicUsize>,
}

impl ActiveRolloutGuard {
    fn new(num_active_rollouts: Arc<AtomicUsize>) -> Self {
        num_active_rollouts.fetch_add(1, Ordering::SeqCst);
        Self {
            num_active_rollouts,
        }
    }
}

impl Drop for ActiveRolloutGuard {
    fn drop(&mut self) {
        let previous = self.num_active_rollouts.fetch_sub(1, Ordering::SeqCst);
        assert!(previous > 0, "num_active_rollouts underflow");
    }
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
) {
    let log_time_progress = |now: Instant| {
        let elapsed_secs = (now - start_time).as_secs_f32().min(total_secs);
        let progress = (elapsed_secs / total_secs).min(1.0);
        let label = format!("Rollout: ({elapsed_secs:.1}s/{total_secs:.1}s)");
        log_master_progress(progress, &label);
    };

    let mut next_progress_log_time = start_time;
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

        let wake_time = std::cmp::min(
            std::cmp::min(next_progress_log_time, deadline),
            Instant::now() + Duration::from_millis(100),
        );
        sleep_until(wake_time).await;
    }

    log_time_progress(Instant::now());
}

async fn wait_for_permit_or_stop(
    question_semaphore: Arc<Semaphore>,
    stop_signal: Arc<AtomicBool>,
) -> Option<OwnedSemaphorePermit> {
    tokio::select! {
        permit = question_semaphore.acquire_owned() => Some(permit.unwrap()),
        _ = async {
            while !stop_signal.load(Ordering::Relaxed) {
                sleep(Duration::from_millis(100)).await;
            }
        } => None,
    }
}

async fn run_rollout_orchestration<M: LlmModelMarker>(ctx: RolloutExecutionContext<M>) {
    let RolloutExecutionContext {
        question_keys,
        dataset,
        rollout_config,
        posterior_calculation_config,
        rollout_store,
        llm_callable,
        client,
        question_semaphore,
        python_tool_pool,
        sglang_waiting_workers,
        stop_signal,
        num_finished_branches,
        num_finished_trees,
        num_correct_branches,
        llm_call_stats,
        num_requeued,
        num_long_running_rollouts,
        num_active_rollouts,
        max_rollout_concurrency,
        max_num_long_running_rollouts,
        total_branches_to_finish,
        total_trees_to_finish,
    } = ctx;

    let handle_rollout_result =
        |result: Result<RolloutOutcome, StopRequestedError>, pending: &mut VecDeque<usize>| {
            match result {
                Ok(RolloutOutcome::Completed) => {}
                Ok(RolloutOutcome::Requeue { flat_id }) => {
                    pending.push_back(flat_id);
                }
                Err(StopRequestedError) => {}
            }
        };

    let mut join_set = JoinSet::new();
    let mut pending_question_keys: VecDeque<usize> = question_keys.into();
    while (!pending_question_keys.is_empty() || !join_set.is_empty())
        && !stop_signal.load(Ordering::Relaxed)
    {
        tokio::select! {
            maybe_owned_permit = wait_for_permit_or_stop(question_semaphore.clone(), stop_signal.clone()), if !pending_question_keys.is_empty() => {
                let Some(owned_permit) = maybe_owned_permit else {
                    break;
                };
                if stop_signal.load(Ordering::Relaxed) {
                    drop(owned_permit);
                    break;
                }

                let question_key = pending_question_keys
                    .pop_front()
                    .expect("queue should not be empty");
                let question = dataset.get(question_key).await.unwrap().unwrap();
                let rollout_config_clone = rollout_config.clone();
                let posterior_calculation_config_clone = posterior_calculation_config.clone();
                let rollout_store = rollout_store.clone();
                let llm_callable_clone = llm_callable.clone();
                let client_clone = client.clone();
                let sglang_waiting_workers_clone = sglang_waiting_workers.clone();
                let python_tool_pool_clone = python_tool_pool.clone();
                let stop_signal_clone = stop_signal.clone();
                let num_finished_branches = num_finished_branches.clone();
                let num_finished_trees = num_finished_trees.clone();
                let num_correct_branches = num_correct_branches.clone();
                let llm_call_stats = llm_call_stats.clone();
                let num_requeued = num_requeued.clone();
                let num_long_running_rollouts = num_long_running_rollouts.clone();
                let num_active_rollouts = num_active_rollouts.clone();
                join_set.spawn(async move {
                    let active_rollouts_for_task = num_active_rollouts.clone();
                    let _active_rollout_guard = ActiveRolloutGuard::new(num_active_rollouts);
                    let result = rollout::<M>(
                        question,
                        rollout_config_clone,
                        posterior_calculation_config_clone,
                        rollout_store,
                        llm_callable_clone,
                        client_clone,
                        python_tool_pool_clone,
                        sglang_waiting_workers_clone,
                        stop_signal_clone,
                        num_finished_branches,
                        num_finished_trees,
                        num_correct_branches,
                        llm_call_stats,
                        num_requeued,
                        num_long_running_rollouts,
                        max_rollout_concurrency,
                        active_rollouts_for_task,
                        max_num_long_running_rollouts,
                        total_branches_to_finish,
                        total_trees_to_finish,
                    )
                    .await;
                    drop(owned_permit);
                    result
                });
            }
            maybe_result = join_set.join_next(), if !join_set.is_empty() => {
                let result = maybe_result.expect("join_set should have at least one task");
                let task_result = result.expect("direct rollout worker task panicked or was cancelled");
                handle_rollout_result(task_result, &mut pending_question_keys);
            }
        }
    }

    while let Some(result) = join_set.join_next().await {
        let task_result = result.expect("direct rollout worker task panicked or was cancelled");
        handle_rollout_result(task_result, &mut pending_question_keys);
    }

    stop_signal.store(true, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy)]
pub struct StopRequestedError;

async fn rollout<M: LlmModelMarker>(
    question: HybridDatasetQuestion,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    rollout_store: SqliteStore<usize, DirectTreeActionLog<M>>,
    llm_callable: M::Callable,
    client: Client,
    python_tool_pool: Arc<PythonToolServerPool>,
    sglang_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
    num_finished_branches: Arc<AtomicUsize>,
    num_finished_trees: Arc<AtomicUsize>,
    num_correct_branches: Arc<AtomicUsize>,
    llm_call_stats: Arc<RwLock<LlmCallStats>>,
    num_requeued: Arc<AtomicUsize>,
    num_long_running_rollouts: Arc<AtomicUsize>,
    max_rollout_concurrency: usize,
    num_active_rollouts: Arc<AtomicUsize>,
    max_num_long_running_rollouts: usize,
    total_branches_to_finish: usize,
    total_trees_to_finish: usize,
) -> Result<RolloutOutcome, StopRequestedError> {
    let mut action_log = rollout_store
        .get(question.flat_id)
        .await
        .unwrap()
        .unwrap_or_else(|| DirectTreeActionLog {
            question: question.clone(),
            rollout_config: rollout_config.clone(),
            posterior_calculation_config: posterior_calculation_config.clone(),
            actions: vec![],
        });
    let mut llm_calls_so_far = action_log
        .actions
        .iter()
        .filter(|action| action_is_llm_call(action))
        .count();
    let mut long_running_guard = LongRunningRolloutGuard::new(num_long_running_rollouts);
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
                let finished = num_finished_branches
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                log_worker_progress(
                    "branches",
                    finished as f32 / total_branches_to_finish as f32,
                    format!(
                        "Num Branches Completed: {}/{}",
                        finished, total_branches_to_finish
                    ),
                );
                let running_accuracy = num_correct as f32 / finished as f32;
                log_key_value_pair("Tree running accuracy", running_accuracy.to_string());
            }
            _ => {}
        }
        if action_is_llm_call(&action) {
            llm_calls_so_far += 1;
        }
        action_log.actions.push(action);
        rollout_store
            .upsert(
                question.flat_id,
                &action_log,
                SqliteBusyRetryConfig::aggressive(),
            )
            .await
            .unwrap();

        if !long_running_guard.has_slot() {
            let q3 = llm_call_stats.read().unwrap().q3();
            if let Some(q3) = q3 {
                if llm_calls_so_far > q3
                    && !long_running_guard.try_acquire_slot(max_num_long_running_rollouts)
                {
                    let current_concurrency = num_active_rollouts.load(Ordering::SeqCst);
                    if (current_concurrency as u128) * 100 <= (max_rollout_concurrency as u128) * 95 {
                        continue;
                    }
                    let new_num_requeued = num_requeued.fetch_add(1, Ordering::SeqCst) + 1;
                    log_key_value_pair("num_requeued", new_num_requeued.to_string());
                    return Ok(RolloutOutcome::Requeue {
                        flat_id: question.flat_id,
                    });
                }
            }
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
        .unwrap()
        .update_and_get_summary(llm_calls_so_far);
    log_key_value_pair("min_llm_calls_per_tree", summary.min.to_string());
    log_key_value_pair("median_llm_calls_per_tree", summary.median.to_string());
    log_key_value_pair("third_quartile_llm_calls_per_tree", summary.q3.to_string());
    log_key_value_pair("max_llm_calls_per_tree", summary.max.to_string());
    Ok(RolloutOutcome::Completed)
}

pub struct RolloutProgramConfig {
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub client: Client,
    pub question_semaphore: Arc<Semaphore>,
    pub max_rollout_concurrency: usize,
    pub llm_cli_args: LlmCliArgs,
    pub rollout_time_limit_secs: usize,
    pub max_sqlite_connections: u32,
    pub num_python_tool_servers: usize,
}

pub async fn rollout_all<M: LlmModelMarker>(program_config: RolloutProgramConfig) {
    let RolloutProgramConfig {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        client,
        question_semaphore,
        max_rollout_concurrency,
        llm_cli_args,
        rollout_time_limit_secs,
        max_sqlite_connections,
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

    let stop_signal = Arc::new(AtomicBool::new(false));
    let python_tool_pool = Arc::new(
        PythonToolServerPool::new(num_python_tool_servers)
            .await
            .expect("failed to initialize python tool server pool"),
    );
    let llm_callable = M::Callable::from_cli_args(client.clone(), &llm_cli_args);
    let asset_file_dataset = AssetFileHybridDataset {
        split: rollout_config.split.clone(),
    };
    let dataset = asset_file_dataset.fetch().await;
    let asset_file_action_logs = AssetFileDirectTreeActionLogs::<M> {
        nickname: config_nickname,
        rollout_config: rollout_config.clone(),
        posterior_calculation_config: posterior_calculation_config.clone(),
        epoch,
        _phantom: std::marker::PhantomData,
    };
    asset_file_action_logs.delete_target_file_if_stale();
    asset_file_action_logs.create_tracking_file();
    let rollout_store = SqliteStore::<usize, DirectTreeActionLog<M>>::initialize_if_missing(
        asset_file_action_logs.file_path(),
        max_sqlite_connections,
    )
    .await;
    let mut question_keys = dataset.get_keys().await.unwrap();
    // sort by question id to ensure deterministic order
    question_keys.sort();
    let sglang_waiting_workers = Arc::new(AtomicUsize::new(0));
    let num_finished_branches = Arc::new(AtomicUsize::new(0));
    let num_finished_trees = Arc::new(AtomicUsize::new(0));
    let num_correct_branches = Arc::new(AtomicUsize::new(0));
    let llm_call_stats = Arc::new(RwLock::new(LlmCallStats::new()));
    let num_requeued = Arc::new(AtomicUsize::new(0));
    let num_long_running_rollouts = Arc::new(AtomicUsize::new(0));
    let num_active_rollouts = Arc::new(AtomicUsize::new(0));
    let max_num_long_running_rollouts = std::cmp::max(1, max_rollout_concurrency / 4);
    let total_branches_to_finish = question_keys.len() * rollout_config.max_num_total_trajectories;
    let total_trees_to_finish = question_keys.len();
    let rollout_context = RolloutExecutionContext::<M> {
        question_keys,
        dataset,
        rollout_config,
        posterior_calculation_config,
        rollout_store,
        llm_callable,
        client,
        question_semaphore,
        python_tool_pool,
        sglang_waiting_workers,
        stop_signal: stop_signal.clone(),
        num_finished_branches,
        num_finished_trees,
        llm_call_stats,
        num_requeued,
        num_long_running_rollouts,
        num_active_rollouts,
        max_rollout_concurrency,
        max_num_long_running_rollouts,
        total_branches_to_finish,
        total_trees_to_finish,
        num_correct_branches,
    };

    tokio::join!(
        run_progress_timer(start_time, deadline, total_secs, stop_signal),
        run_rollout_orchestration::<M>(rollout_context)
    );
    // restore the progress bars
    delete_worker_progress_bar("branches");
    delete_worker_progress_bar("trees");
    log_master_progress(1.0, "Rollout: time up or all finished");
}
