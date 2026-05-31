use std::sync::Arc;

use rayon::{ThreadPoolBuilder, prelude::*};
use research_utility::{asset_file::AssetFile, log_message::log_master_progress};

use crate::{
    direct_tool::{
        direct_tree::DirectTree,
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        direct_tree_advantage::WinRate,
    },
    llm_model::LlmModelMarker,
};

fn question_is_correct<M: LlmModelMarker>(action_log: &DirectTreeActionLog<M>) -> Option<bool> {
    let tree = DirectTree::<M>::from_action_log(action_log);
    assert!(
        tree.leaf_segment_judgments.len() <= 1,
        "There should be at most one leaf segment judgment for accuracy calculation"
    );
    tree.leaf_segment_judgments
        .values()
        .next()
        .map(|judgment| judgment.is_correct)
}

pub async fn read_accuracy<M: LlmModelMarker>(
    asset_file_action_logs: AssetFileDirectTreeActionLogs<M>,
) -> WinRate {
    let action_logs_store = asset_file_action_logs.fetch().await;
    let mut keys = action_logs_store.get_keys().await.unwrap();
    keys.sort();

    let nproc = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let num_worker_threads = std::cmp::max((nproc + 1) / 2, 1);
    let worker_pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(num_worker_threads)
            .build()
            .expect("Failed to build rayon worker pool for accuracy calculation"),
    );

    log_master_progress(0.0, "Accuracy: Calculating");

    let num_keys = keys.len();
    let chunk_size = std::cmp::max(num_worker_threads, 1);
    let mut num_wins = 0usize;
    let mut total_plays = 0usize;
    let mut finished = 0usize;

    for chunk_keys in keys.chunks(chunk_size) {
        let mut chunk: Vec<DirectTreeActionLog<M>> = Vec::with_capacity(chunk_keys.len());
        for key in chunk_keys {
            chunk.push(
                action_logs_store
                    .get(*key)
                    .await
                    .unwrap()
                    .expect("key from sqlite key set must exist"),
            );
        }

        let pool = worker_pool.clone();
        let chunk_results: Vec<Option<bool>> = tokio::task::spawn_blocking(move || {
            pool.install(|| {
                chunk
                    .into_par_iter()
                    .map(|action_log| question_is_correct::<M>(&action_log))
                    .collect()
            })
        })
        .await
        .expect("blocking task for accuracy calculation panicked");

        for result in chunk_results {
            if let Some(is_correct) = result {
                if is_correct {
                    num_wins += 1;
                }
                total_plays += 1;
            }
        }
        finished += chunk_keys.len();
        let progress = finished as f32 / num_keys as f32;
        log_master_progress(progress, "Accuracy: Calculating");
    }

    log_master_progress(1.0, "Accuracy: Done");

    WinRate {
        num_wins,
        total_plays,
    }
}
