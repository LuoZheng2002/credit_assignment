use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use reqwest::Client;
use research_utility::{
    asset_file::AssetFile,
    log_message::{delete_worker_progress_bar, log_master_progress, log_worker_progress},
    sqlite_store::{SqliteBusyRetryConfig, SqliteStore},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep, sleep_until};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::DirectTree,
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
    total_branches_to_finish: usize,
    total_trees_to_finish: usize,
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
        total_branches_to_finish,
        total_trees_to_finish,
    } = ctx;

    let mut join_set = JoinSet::new();
    for question_key in question_keys {
        if stop_signal.load(Ordering::Relaxed) {
            break;
        }

        let maybe_owned_permit =
            wait_for_permit_or_stop(question_semaphore.clone(), stop_signal.clone()).await;
        let Some(owned_permit) = maybe_owned_permit else {
            break;
        };

        if stop_signal.load(Ordering::Relaxed) {
            drop(owned_permit);
            break;
        }

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
        join_set.spawn(async move {
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
                total_branches_to_finish,
                total_trees_to_finish,
            )
            .await;
            // we do not care about whether the rollout is interrupted by stop signal
            let _ = result;
            drop(owned_permit);
        });
    }

    while let Some(result) = join_set.join_next().await {
        result.expect("direct rollout worker task panicked or was cancelled");
    }

    stop_signal.store(true, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy)]
pub struct StopRequestedError;

pub async fn rollout<M: LlmModelMarker>(
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
    total_branches_to_finish: usize,
    total_trees_to_finish: usize,
) -> Result<(), StopRequestedError> {
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
    loop {
        let tree = DirectTree::<M>::from_action_log(&action_log);
        if tree.completed {
            break;
        }
        let new_actions = tree
            .produce_actions_from_direct_tree(
                &llm_callable,
                client.clone(),
                python_tool_pool.clone(),
                sglang_waiting_workers.clone(),
                stop_signal.clone(),
            )
            .await?;
        if new_actions.is_empty() {
            break;
        }
        for action in new_actions {
            if matches!(
                action,
                DirectTreeAction::CreateAndJudgeTrunkTrajectory { .. }
                    | DirectTreeAction::CreateAndJudgeBranchSegment { .. }
            ) {
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
            }
            action_log.actions.push(action);
        }
        rollout_store
            .upsert(
                question.flat_id,
                &action_log,
                SqliteBusyRetryConfig::aggressive(),
            )
            .await
            .unwrap();
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
    Ok(())
}

pub struct RolloutProgramConfig {
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize, // the epoch index
    pub client: Client,
    pub question_semaphore: Arc<Semaphore>,
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
        total_branches_to_finish,
        total_trees_to_finish,
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
