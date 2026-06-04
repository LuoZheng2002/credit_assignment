use research_utility::{asset_file::AssetFile, progress_tui_server::log_master_progress};
use std::{marker::PhantomData, sync::Arc};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    direct_tool::{
        direct_tree::DirectTree,
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        hybrid_dataset::{AssetFileHybridDataset, DatasetSplit},
    },
    llm_model::LlmModelMarker,
};

#[derive(Debug, Clone)]
pub struct AccuracyStats {
    pub weighted_num_wins: f32,
    pub weighted_total_plays: f32,
    pub num_trees_with_judgments: usize,
    pub num_trajectories_judged: usize,
}

impl AccuracyStats {
    pub fn accuracy(&self) -> Option<f32> {
        if self.weighted_total_plays == 0.0 {
            None
        } else {
            Some(self.weighted_num_wins / self.weighted_total_plays)
        }
    }
}

fn tree_accuracy<M: LlmModelMarker, S: DatasetSplit>(
    action_log: &DirectTreeActionLog<M, S>,
) -> Option<(usize, usize)> {
    let tree = DirectTree::<M, S>::from_action_log(action_log);
    let total_trajectories = tree.leaf_segment_judgments.len();
    if total_trajectories == 0 {
        return None;
    }
    let num_correct_trajectories = tree
        .leaf_segment_judgments
        .values()
        .filter(|judgment| judgment.is_correct)
        .count();
    Some((num_correct_trajectories, total_trajectories))
}

pub async fn get_accuracy<M: LlmModelMarker, S: DatasetSplit>(
    asset_file_action_logs: AssetFileDirectTreeActionLogs<M, S>,
    progress_bar_label: &str,
) -> AccuracyStats {
    let asset_file_dataset = AssetFileHybridDataset::<S>(PhantomData);
    let question_store = asset_file_dataset.fetch().await;
    let action_store = asset_file_action_logs.fetch().await;
    let mut keys = action_store.get_keys().unwrap();
    keys.sort();

    log_master_progress(0.0, format!("{}: Calculating", progress_bar_label));

    let num_keys = keys.len();
    let mut weighted_num_wins = 0.0f32;
    let mut weighted_total_plays = 0.0f32;
    let mut num_trees_with_judgments = 0usize;
    let mut num_trajectories_judged = 0usize;

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
                    tree_accuracy::<M, S>(&action_log)
                });
            }
            joined = join_set.join_next(), if !join_set.is_empty() => {
                finished += 1;

                match joined.expect("join_set must have at least one task") {
                    Ok(result) => {
                        if let Some((num_correct_trajectories, total_trajectories)) = result {
                            weighted_num_wins +=
                                num_correct_trajectories as f32 / total_trajectories as f32;
                            weighted_total_plays += 1.0;
                            num_trees_with_judgments += 1;
                            num_trajectories_judged += total_trajectories;
                        }
                    }
                    Err(join_err) => panic!("accuracy task panicked: {join_err}"),
                }

                let progress = if num_keys == 0 {
                    1.0
                } else {
                    finished as f32 / num_keys as f32
                };
                log_master_progress(progress, format!("{}: Calculating", progress_bar_label));
            }
        }
    }

    log_master_progress(1.0, format!("{}: Done", progress_bar_label));

    AccuracyStats {
        weighted_num_wins,
        weighted_total_plays,
        num_trees_with_judgments,
        num_trajectories_judged,
    }
}
