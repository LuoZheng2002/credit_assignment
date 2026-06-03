use futures::stream::{self, StreamExt};
use research_utility::{asset_file::AssetFile, progress_tui_server::log_master_progress};

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

    log_master_progress(0.0, "Accuracy: Calculating");

    let num_keys = keys.len();
    let mut num_wins = 0usize;
    let mut total_plays = 0usize;

    const MAX_CONCURRENT_TASKS: usize = 200;
    let mut result_stream = stream::iter(keys.into_iter())
        .map(|key| {
            let action_logs_store = action_logs_store.clone();
            async move {
                let action_log = action_logs_store
                    .get(key)
                    .await
                    .unwrap()
                    .expect("key from sqlite key set must exist");
                question_is_correct::<M>(&action_log)
            }
        })
        .buffer_unordered(MAX_CONCURRENT_TASKS);

    let mut finished = 0usize;
    while let Some(result) = result_stream.next().await {
        finished += 1;
        if let Some(is_correct) = result {
            if is_correct {
                num_wins += 1;
            }
            total_plays += 1;
        }
        let progress = if num_keys == 0 {
            1.0
        } else {
            finished as f32 / num_keys as f32
        };
        log_master_progress(progress, "Accuracy: Calculating");
    }

    log_master_progress(1.0, "Accuracy: Done");

    WinRate {
        num_wins,
        total_plays,
    }
}
