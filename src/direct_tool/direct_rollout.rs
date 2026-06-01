use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use futures::StreamExt;
use kll_rs::KllFloatSketch;
use reqwest::Client;
use research_utility::{
    asset_file::AssetFile,
    log_message::{
        delete_worker_progress_bar, log_key_value_pair, log_master_progress, log_worker_progress,
    },
};
use tokio::sync::{RwLock, mpsc};
use tokio::time::{Duration, Instant, sleep_until};
use tokio_stream::wrappers::ReceiverStream;

use crate::atomic_count_guard::AtomicCountGuard;
use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::{DirectTree, SegmentContent},
        direct_tree_action::DirectTreeAction,
        direct_tree_action_log::{
            AssetFileDirectTreeActionLogs, DirectTreeActionLogMetadata,
            DirectTreeActionLogStore,
        },
        hybrid_dataset::{AssetFileHybridDataset, HybridDatasetQuestion},
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    llm_model::{LlmCallable, LlmCliArgs, LlmModelMarker},
    tool_call_python::PythonToolServerPool,
};

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
    newly_finished_trees: Arc<AtomicUsize>,
    num_correct_branches: Arc<AtomicUsize>,
    llm_call_stats: Arc<RwLock<LlmCallStats>>,
    num_requeued: Arc<AtomicUsize>,
    num_active_rollouts: Arc<AtomicUsize>,
    num_active_long_running_rollouts: Arc<AtomicUsize>,
    max_rollout_concurrency: usize,
    max_num_long_running_rollouts: usize,
    total_branches_to_finish: usize,
    total_trees_to_finish: usize,
}

const NUM_WARMUP_TREES: usize = 200;

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
) -> Result<RolloutOutcome, StopRequestedError> {
    let RolloutSharedStates {
        python_tool_pool,
        sglang_waiting_workers,
        judge_waiting_workers,
        tool_waiting_workers,
        sqlite_waiting_workers,
        stop_signal,
        num_finished_branches,
        num_finished_trees,
        newly_finished_trees,
        num_correct_branches,
        llm_call_stats,
        num_requeued,
        num_active_rollouts,
        num_active_long_running_rollouts,
        max_rollout_concurrency,
        max_num_long_running_rollouts,
        total_branches_to_finish,
        total_trees_to_finish,
    } = shared_states;

    let _active_rollouts_guard =
        AtomicCountGuard::new(num_active_rollouts.clone(), "num_active_rollouts");
    {
        let _sqlite_waiting_workers_guard = AtomicCountGuard::new(
            sqlite_waiting_workers.clone(),
            "sqlite_waiting_workers",
        );
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
        let _sqlite_waiting_workers_guard = AtomicCountGuard::new(
            sqlite_waiting_workers.clone(),
            "sqlite_waiting_workers",
        );
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
    let mut long_running_guard: Option<AtomicCountGuard> = None;
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
        }
        let newest_action_index = action_log.actions.len();
        action_log.actions.push(action);
        {
            let _sqlite_waiting_workers_guard = AtomicCountGuard::new(
                sqlite_waiting_workers.clone(),
                "sqlite_waiting_workers",
            );
            rollout_store
                .append_action_at(
                    question.flat_id,
                    newest_action_index,
                    action_log.actions.last().unwrap(),
                )
                .await
                .unwrap();
        }

        if long_running_guard.is_none() {
            let q3 = llm_call_stats.read().await.q3();
            if let Some(q3) = q3 {
                if llm_calls_so_far > q3 {
                    match AtomicCountGuard::try_new_with_max(
                        num_active_long_running_rollouts.clone(),
                        "num_active_long_running_rollouts",
                        max_num_long_running_rollouts,
                    ) {
                        Some(guard) => {
                            long_running_guard = Some(guard);
                        }
                        None => {
                            let current_concurrency = num_active_rollouts.load(Ordering::SeqCst);
                            if (current_concurrency as u128) * 100
                                <= (max_rollout_concurrency as u128) * 95
                            {
                                continue;
                            }
                            let newly_finished_trees = newly_finished_trees.load(Ordering::SeqCst);
                            if newly_finished_trees < NUM_WARMUP_TREES {
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
        }
    }
    // log_info(format!("Rollout {} finished", question.flat_id));
    drop(long_running_guard.take());
    let finished = num_finished_trees.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    newly_finished_trees.fetch_add(1, Ordering::SeqCst);
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
        .await
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
    let rollout_store = DirectTreeActionLogStore::<M>::initialize_if_missing(
        asset_file_action_logs.file_path(),
        max_sqlite_connections,
    )
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
    let newly_finished_trees = Arc::new(AtomicUsize::new(0));
    let num_correct_branches = Arc::new(AtomicUsize::new(0));
    let num_active_rollouts = Arc::new(AtomicUsize::new(0));
    let llm_call_stats = Arc::new(RwLock::new(LlmCallStats::new()));
    let num_requeued = Arc::new(AtomicUsize::new(0));
    let max_num_long_running_rollouts = std::cmp::max(1, max_rollout_concurrency / 4);
    let num_active_long_running_rollouts = Arc::new(AtomicUsize::new(0));
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
        newly_finished_trees,
        num_correct_branches,
        llm_call_stats,
        num_requeued,
        num_active_rollouts,
        num_active_long_running_rollouts,
        max_rollout_concurrency,
        max_num_long_running_rollouts,
        total_branches_to_finish,
        total_trees_to_finish,
    };

    let progress_timer_handle = tokio::spawn(run_progress_timer(
        start_time,
        deadline,
        total_secs,
        shared_states.stop_signal.clone(),
    ));

    let queue_capacity = std::cmp::max(1, max_rollout_concurrency * 4);
    let (queue_tx, queue_rx) = mpsc::channel::<usize>(queue_capacity);
    for question_key in question_keys {
        queue_tx
            .send(question_key)
            .await
            .expect("initial rollout queue send should not fail");
    }

    let stop_signal_for_take_while = shared_states.stop_signal.clone();
    let result_stream = ReceiverStream::new(queue_rx)
        .take_while(move |_| {
            let stop_signal = stop_signal_for_take_while.clone();
            async move { !stop_signal.load(Ordering::Relaxed) }
        })
        .map(|question_key| {
            let dataset = dataset.clone();
            let rollout_config_clone = rollout_config.clone();
            let posterior_calculation_config_clone = posterior_calculation_config.clone();
            let rollout_store = rollout_store.clone();
            let llm_callable_clone = llm_callable.clone();
            let client_clone = client.clone();
            let shared_states_clone = shared_states.clone();
            async move {
                let question = dataset
                    .get(question_key)
                    .await
                    .unwrap()
                    .expect("question key from rollout queue must exist");
                rollout::<M>(
                    question,
                    rollout_config_clone,
                    posterior_calculation_config_clone,
                    rollout_store,
                    llm_callable_clone,
                    client_clone,
                    shared_states_clone,
                )
                .await
            }
        })
        .buffer_unordered(max_rollout_concurrency);
    futures::pin_mut!(result_stream);

    let mut num_pending_questions = total_trees_to_finish;
    while let Some(task_result) = result_stream.next().await {
        match task_result {
            Ok(RolloutOutcome::Completed) | Err(StopRequestedError) => {
                num_pending_questions = num_pending_questions.saturating_sub(1);
            }
            Ok(RolloutOutcome::Requeue { flat_id }) => {
                if shared_states.stop_signal.load(Ordering::Relaxed) {
                    num_pending_questions = num_pending_questions.saturating_sub(1);
                } else {
                    queue_tx
                        .send(flat_id)
                        .await
                        .expect("requeue send should not fail while stream is active");
                }
            }
        }
        if num_pending_questions == 0 {
            break;
        }
    }

    drop(queue_tx);

    shared_states.stop_signal.store(true, Ordering::Relaxed);
    progress_timer_handle
        .await
        .expect("progress timer task panicked");
    // restore the progress bars
    delete_worker_progress_bar("branches");
    delete_worker_progress_bar("trees");
    log_master_progress(1.0, "Rollout: time up or all finished");
}
