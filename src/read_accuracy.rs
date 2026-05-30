use research_utility::asset_file::AssetFile;

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
        tree.completed,
        "Direct tree must be completed before accuracy calculation"
    );
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

    let mut num_wins = 0usize;
    let mut total_plays = 0usize;
    for key in keys {
        let action_log = action_logs_store
            .get(key)
            .await
            .unwrap()
            .expect("key from sqlite key set must exist");
        if let Some(is_correct) = question_is_correct::<M>(&action_log) {
            if is_correct {
                num_wins += 1;
            }
            total_plays += 1;
        }
    }

    WinRate {
        num_wins,
        total_plays,
    }
}
