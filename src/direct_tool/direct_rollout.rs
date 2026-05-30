use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use reqwest::Client;
use research_utility::{
    asset_file::AssetFile,
    log_message::{log_info, log_master_progress},
    sqlite_store::{SqliteBusyRetryConfig, SqliteStore},
};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio::time::{Duration, Instant, sleep_until};

use crate::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::DirectTree,
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        hybrid_dataset::{AssetFileHybridDataset, HybridDatasetQuestion},
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    llm_model::{LlmCallable, LlmCliArgs, LlmModelMarker},
};

pub async fn rollout<M: LlmModelMarker>(
    question: HybridDatasetQuestion,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    rollout_store: SqliteStore<usize, DirectTreeActionLog<M>>,
    llm_callable: M::Callable,
    client: Client,
    sglang_waiting_workers: Arc<AtomicUsize>,
    stop_signal: Arc<AtomicBool>,
    // rng: &mut StdRng,
) {
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
    // log_key_value_pair(
    //     "status".to_string(),
    //     format!(
    //         "Loading {} existing actions for question flat id {}...",
    //         action_log.actions.len(),
    //         question.flat_id
    //     ),
    // );
    log_info(format!(
        "Loading {} existing actions for question flat id {}...",
        action_log.actions.len(),
        question.flat_id
    ));
    loop {
        if stop_signal.load(Ordering::Relaxed) {
            break;
        }
        let tree = DirectTree::<M>::from_action_log(&action_log);
        if tree.completed {
            break;
        }
        let new_actions = tree
            .produce_actions_from_direct_tree(
                &llm_callable,
                client.clone(),
                sglang_waiting_workers.clone(),
                stop_signal.clone(),
            )
            .await;
        if new_actions.is_empty() {
            break;
        }
        for action in new_actions {
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
    log_info(format!("Rollout {} finished", question.flat_id));
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
    } = program_config;
    assert!(
        rollout_time_limit_secs > 0,
        "rollout_time_limit_secs must be positive"
    );
    let rollout_time_limit = Duration::from_secs(rollout_time_limit_secs as u64);
    let start_time = Instant::now();
    let deadline = start_time + rollout_time_limit;
    let total_secs = rollout_time_limit_secs as f32;

    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_for_progress = stop_signal.clone();
    let progress_task = tokio::spawn(async move {
        let log_time_progress = |now: Instant| {
            let elapsed_secs = (now - start_time).as_secs_f32().min(total_secs);
            let progress = (elapsed_secs / total_secs).min(1.0);
            let label = format!("Rollout: ({elapsed_secs:.1}s/{total_secs:.1}s)");
            log_master_progress(progress, &label);
        };

        let mut next_progress_log_time = start_time;
        loop {
            if stop_signal_for_progress.load(Ordering::Relaxed) {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                stop_signal_for_progress.store(true, Ordering::Relaxed);
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
    });
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
    let mut join_set = JoinSet::new();
    let sglang_waiting_workers = Arc::new(AtomicUsize::new(0));
    let num_keys = question_keys.len();
    let mut num_finished = 0;
    for question_key in question_keys {
        let now = Instant::now();
        if now >= deadline {
            stop_signal.store(true, Ordering::Relaxed);
            break;
        }

        let owned_permit = tokio::select! {
            permit = question_semaphore.clone().acquire_owned() => permit.unwrap(),
            _ = sleep_until(deadline) => {
                stop_signal.store(true, Ordering::Relaxed);
                break;
            }
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
        let stop_signal_clone = stop_signal.clone();
        join_set.spawn(async move {
            rollout::<M>(
                question,
                rollout_config_clone,
                posterior_calculation_config_clone,
                rollout_store,
                llm_callable_clone,
                client_clone,
                sglang_waiting_workers_clone,
                stop_signal_clone,
            )
            .await;
            drop(owned_permit);
        });

        while let Some(result) = join_set.try_join_next() {
            result.expect("direct rollout worker task panicked or was cancelled");
            num_finished += 1;
            log_info(format!(
                "Progress: {num_finished}/{num_keys} questions finished"
            ));
        }
    }

    while !join_set.is_empty() {
        tokio::select! {
            result = join_set.join_next() => {
                if let Some(result) = result {
                    result.expect("direct rollout worker task panicked or was cancelled");
                }
            }
            _ = sleep_until(deadline), if !stop_signal.load(Ordering::Relaxed) => {
                stop_signal.store(true, Ordering::Relaxed);
            }
        }
    }
    stop_signal.store(true, Ordering::Relaxed);
    progress_task
        .await
        .expect("rollout progress task panicked or was cancelled");
}
