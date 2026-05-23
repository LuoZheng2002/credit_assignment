use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use clap::ValueEnum;
use indexmap::IndexMap;
use pyo3::Python;
use rand::{Rng, SeedableRng, rngs::StdRng};
use reqwest::Client;

use crate::{
    agent::{
        action_log_schema::AssetFileActionLogs,
        rollout_loop::rollout,
        single_dataset::{AssetFileSingleDataset, SingleDatasetQuestion},
        tree_reconstruction::{is_completed, reconstruct_tree},
    },
    asset_file::AssetFile,
    llm_model::{LlmModelMarker, LlmModelName},
    worker_message_tx::{log_key_value_pair, log_master_progress},
};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

fn publish_master_progress(processed_questions: usize, total_questions: usize) {
    let progress = if total_questions == 0 {
        1.0
    } else {
        processed_questions as f32 / total_questions as f32
    };
    log_master_progress(
        progress,
        format!(
            "{}/{} questions processed",
            processed_questions, total_questions
        ),
    );
}

// this function is responsible for loading the dataset, running rollouts and then storing the trajectories.
pub async fn rollout_batch<M: LlmModelMarker + 'static>(
    dataset_name: String,
    num_samples: usize,
    llm_callable: M::Callable,
) {
    let model_name = M::CLI_NAME;
    log_key_value_pair(
        "status".to_string(),
        "Initializing rollout batch".to_string(),
    );
    log_key_value_pair("model".to_string(), model_name.to_string());
    log_key_value_pair("dataset".to_string(), dataset_name.clone());
    log_key_value_pair("num_samples".to_string(), num_samples.to_string());
    assert!(
        std::env::var("PYTHONPATH").is_ok(),
        "PYTHONPATH environment variable is not set"
    );
    Python::initialize();

    let client = Client::new();
    let mut rng = StdRng::seed_from_u64(42);

    let asset_file_action_logs = AssetFileActionLogs {
        model: LlmModelName::from_str(M::CLI_NAME, true).unwrap(),
        dataset: dataset_name.clone(),
        num_samples,
    };
    let rollout_store = asset_file_action_logs.open_store().await;

    let asset_file_dataset = AssetFileSingleDataset {
        dataset: dataset_name.clone(),
        num_samples,
    };
    let dataset = asset_file_dataset.fetch().await;
    log_key_value_pair("status".to_string(), "Dataset loaded".to_string());

    let questions = dataset
        .into_iter()
        .map(|q| (q.id, q))
        .collect::<IndexMap<usize, SingleDatasetQuestion>>();
    let total_questions = questions.len();

    let mut tree_completed_ids: BTreeSet<usize> = BTreeSet::new();
    for (id, question) in &questions {
        if let Some(log) = rollout_store.get(*id).await.unwrap() {
            assert_eq!(
                log.question.id, *id,
                "TreeActionLog.question.id must match sqlite key"
            );
            assert_eq!(
                log.question.question, question.question,
                "TreeActionLog.question.question must remain immutable"
            );
            assert_eq!(
                log.question.final_answer, question.final_answer,
                "TreeActionLog.question.final_answer must remain immutable"
            );
            if is_completed(&log) {
                tree_completed_ids.insert(*id);
            }
        }
    }

    let mut unfinished_tree_ids: BTreeSet<usize> = questions.keys().cloned().collect();
    unfinished_tree_ids.retain(|id| !tree_completed_ids.contains(id));

    log_key_value_pair(
        "status".to_string(),
        format!(
            "Running rollouts for {} unfinished questions",
            unfinished_tree_ids.len()
        ),
    );

    SHUTDOWN.store(false, Ordering::SeqCst);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!("Ctrl+C received, shutting down...");
        SHUTDOWN.store(true, Ordering::SeqCst);
    });

    let mut submit_task_rng = StdRng::seed_from_u64(rng.next_u64());
    let processed_question_count = Arc::new(AtomicUsize::new(tree_completed_ids.len()));
    publish_master_progress(tree_completed_ids.len(), total_questions);

    let mut task_handles = Vec::new();
    for id in unfinished_tree_ids {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        let question = questions.get(&id).cloned().unwrap();
        let store = rollout_store.clone();
        let client = client.clone();
        let llm_callable = llm_callable.clone();
        let mut task_rng = StdRng::seed_from_u64(submit_task_rng.next_u64());
        let processed_question_count = processed_question_count.clone();
        task_handles.push(tokio::spawn(async move {
            rollout::<M>(question, store, llm_callable, client, &mut task_rng).await;
            let processed = processed_question_count.fetch_add(1, Ordering::SeqCst) + 1;
            publish_master_progress(processed, total_questions);
        }));
    }

    for task_handle in task_handles {
        task_handle.await.unwrap();
    }

    log_key_value_pair("status".to_string(), "Finalizing summary".to_string());

    let mut overall_correct = 0usize;
    let mut overall_denominator = 0usize;
    let mut incomplete_ids = Vec::new();
    for id in questions.keys() {
        let maybe_log = rollout_store.get(*id).await.unwrap();
        match maybe_log {
            Some(log) => {
                let tree = reconstruct_tree(&log);
                if tree.completed {
                    overall_correct += tree.correctness_ratio.numerator;
                    overall_denominator += tree.correctness_ratio.denominator;
                } else {
                    incomplete_ids.push(*id);
                }
            }
            None => {
                incomplete_ids.push(*id);
            }
        }
    }

    if !incomplete_ids.is_empty() {
        log_key_value_pair(
            "warning".to_string(),
            format!(
                "Excluding {} incomplete logs from aggregate accuracy (diagnostics only)",
                incomplete_ids.len()
            ),
        );
    }

    let accuracy = if overall_denominator == 0 {
        0.0
    } else {
        overall_correct as f64 / overall_denominator as f64
    };
    log_key_value_pair(
        "status".to_string(),
        format!(
            "Rollout evaluation completed. Overall tree correctness accuracy: {}/{} ({:.2}%)",
            overall_correct,
            overall_denominator,
            accuracy * 100.0
        ),
    );
    publish_master_progress(total_questions - incomplete_ids.len(), total_questions);
    log_key_value_pair("status".to_string(), "Completed".to_string());
}
