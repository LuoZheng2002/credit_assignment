use research_utility::{asset_file::AssetFile, progress_tui_server::log_master_progress};
use std::{marker::PhantomData, sync::Arc};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    direct_tool::{
        direct_tree::DirectTree,
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        direct_tree_advantage::WinRate,
        hybrid_dataset::{AssetFileHybridDataset, Validation},
    },
    llm_model::LlmModelMarker,
};

fn question_is_correct<M: LlmModelMarker>(
    action_log: &DirectTreeActionLog<M, Validation>,
) -> Option<bool> {
    let tree = DirectTree::<M, Validation>::from_action_log(action_log);
    assert!(
        tree.leaf_segment_judgments.len() <= 1,
        "There should be at most one leaf segment judgment for accuracy calculation"
    );
    tree.leaf_segment_judgments
        .values()
        .next()
        .map(|judgment| judgment.is_correct)
}

pub async fn get_validation_accuracy<M: LlmModelMarker>(
    asset_file_action_logs: AssetFileDirectTreeActionLogs<M, Validation>,
) -> WinRate {
    let asset_file_dataset = AssetFileHybridDataset::<Validation>(PhantomData);
    let question_store = asset_file_dataset.fetch().await;
    let action_store = asset_file_action_logs.fetch().await;
    let mut keys = action_store.get_keys().unwrap();
    keys.sort();

    log_master_progress(0.0, "Accuracy: Calculating");

    let num_keys = keys.len();
    let mut num_wins = 0usize;
    let mut total_plays = 0usize;

    const MAX_CONCURRENT_TASKS: usize = 200;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_TASKS));
    let mut join_set = JoinSet::new();
    let mut next_key_index = 0usize;

    let mut finished = 0usize;
    while next_key_index < keys.len() || !join_set.is_empty() {
        tokio::select! {
            permit_result = semaphore.clone().acquire_owned(), if next_key_index < keys.len() => {
                let permit = permit_result.expect("accuracy semaphore should not be closed");
                let key = keys[next_key_index];
                next_key_index += 1;

                let question = question_store.get(key).unwrap().unwrap();
                let actions = action_store.load_table_sorted(key).unwrap();
                let action_log = DirectTreeActionLog {
                    question,
                    rollout_config: asset_file_action_logs.rollout_config.clone(),
                    posterior_calculation_config: asset_file_action_logs
                        .posterior_calculation_config
                        .clone(),
                    actions,
                };

                join_set.spawn(async move {
                    let _permit = permit;
                    question_is_correct::<M>(&action_log)
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                finished += 1;

                match joined.expect("join_set must have at least one task") {
                    Ok(result) => {
                        if let Some(is_correct) = result {
                            if is_correct {
                                num_wins += 1;
                            }
                            total_plays += 1;
                        }
                    }
                    Err(join_err) => panic!("accuracy task panicked: {join_err}"),
                }

                let progress = if num_keys == 0 {
                    1.0
                } else {
                    finished as f32 / num_keys as f32
                };
                log_master_progress(progress, "Accuracy: Calculating");
            }
        }
    }

    log_master_progress(1.0, "Accuracy: Done");

    WinRate {
        num_wins,
        total_plays,
    }
}
