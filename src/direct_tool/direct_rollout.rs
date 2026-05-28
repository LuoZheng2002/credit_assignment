use std::sync::Arc;

use reqwest::Client;
use research_utility::{
    asset_file::AssetFile,
    sqlite_store::{SqliteBusyRetryConfig, SqliteStore},
    worker_message_tx::log_key_value_pair,
};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

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
    log_key_value_pair(
        "status".to_string(),
        format!(
            "Loading {} existing actions for question flat id {}...",
            action_log.actions.len(),
            question.flat_id
        ),
    );
    loop {
        let tree = DirectTree::<M>::from_action_log(&action_log);
        if tree.completed {
            break;
        }
        let new_actions = tree
            .produce_actions_from_direct_tree(&llm_callable, client.clone())
            .await;
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
    log_key_value_pair(
        "info".to_string(),
        format!("Rollout {} finished", question.flat_id),
    );
}

pub struct RolloutProgramConfig {
    pub config_nickname: String,
    pub rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub epoch: usize,
    pub client: Client,
    pub question_semaphore: Arc<Semaphore>,
    pub llm_cli_args: LlmCliArgs,
    pub first_n_samples: Option<usize>,
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
        first_n_samples,
        max_sqlite_connections,
    } = program_config;
    let llm_callable = M::Callable::from_cli_args(client.clone(), &llm_cli_args);
    let asset_file_dataset = AssetFileHybridDataset;
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
    if let Some(first_n) = first_n_samples {
        question_keys.truncate(first_n);
    }
    let mut join_set = JoinSet::new();
    let num_keys = question_keys.len();
    let mut num_finished = 0;
    for question_key in question_keys {
        let owned_permit = question_semaphore.clone().acquire_owned().await.unwrap();
        let question = dataset.get(question_key).await.unwrap().unwrap();
        let rollout_config_clone = rollout_config.clone();
        let posterior_calculation_config_clone = posterior_calculation_config.clone();
        let rollout_store = rollout_store.clone();
        let llm_callable_clone = llm_callable.clone();
        let client_clone = client.clone();
        join_set.spawn(async move {
            rollout::<M>(
                question,
                rollout_config_clone,
                posterior_calculation_config_clone,
                rollout_store,
                llm_callable_clone,
                client_clone,
            )
            .await;
            drop(owned_permit);
        });

        while let Some(result) = join_set.try_join_next() {
            result.expect("direct rollout worker task panicked or was cancelled");
            num_finished += 1;
            log_key_value_pair(
                "progress".to_string(),
                format!("{num_finished}/{num_keys} questions finished"),
            );
        }
    }

    while let Some(result) = join_set.join_next().await {
        result.expect("direct rollout worker task panicked or was cancelled");
    }
}
