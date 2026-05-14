use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use futures::{
    future::join_all,
    stream::{FuturesUnordered, StreamExt},
};
use indexmap::IndexMap;
use pyo3::Python;
use rand::{Rng, SeedableRng, rngs::StdRng};
use reqwest::Client;
use tokio::sync::OwnedSemaphorePermit;

use crate::{
    agent::{
        rollout_loop::rollout,
        sqlite_rollout_log::{SqliteSessionLogStore, get_rollout_log_path},
        tree_action::TreeAction,
        tree_schema::{AssetFileTrees, AssetFileTreesTracking, CompletedTree, CompletedTreeStore},
    },
    call_llm::LlmEndpoint,
    datasets::{AssetFileDataset, DeepMathQuestion},
    direct_answer::generate_raw_answers::LlmModel,
    parallel_process_jsonl::write_json,
    version_tracking::AssetFile,
};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
const MAX_CONCURRENT_REQUESTS_PER_ENDPOINT: usize = 100;

async fn choose_endpoint_for_question(
    llm_endpoints: &[Arc<LlmEndpoint>],
) -> (Arc<LlmEndpoint>, OwnedSemaphorePermit) {
    assert!(!llm_endpoints.is_empty(), "llm_endpoints must not be empty");

    let (most_available_endpoint_idx, max_available_slots) = llm_endpoints
        .iter()
        .enumerate()
        .max_by_key(|(_, endpoint)| endpoint.question_slot_semaphore.available_permits())
        .expect("llm_endpoints must not be empty");

    if max_available_slots.question_slot_semaphore.available_permits() > 0 {
        let endpoint = llm_endpoints[most_available_endpoint_idx].clone();
        let permit = endpoint
            .question_slot_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        return (endpoint, permit);
    }

    let mut acquire_futures = FuturesUnordered::new();
    for endpoint in llm_endpoints.iter().cloned() {
        acquire_futures.push(async move {
            let permit = endpoint
                .question_slot_semaphore
                .clone()
                .acquire_owned()
                .await
                .unwrap();
            (endpoint, permit)
        });
    }
    acquire_futures
        .next()
        .await
        .expect("llm_endpoints must not be empty")
}

// this function is responsible for loading the dataset, running rollouts and then storing the trajectories.
// It is also responsible for creating the version tracking file and output file if they do not exist
pub async fn rollout_batch(
    model: LlmModel,
    dataset_name: String,
    num_samples: usize,
    vllm_ports: Vec<u16>,
) {
    assert!(!vllm_ports.is_empty(), "--vllm-ports must not be empty");
    assert!(
        vllm_ports.iter().all(|port| *port > 0),
        "all --vllm-ports must be greater than 0"
    );

    let llm_endpoints: Vec<Arc<LlmEndpoint>> = vllm_ports
        .iter()
        .enumerate()
        .map(|(id, port)| {
            Arc::new(LlmEndpoint::new(
                id,
                *port,
                MAX_CONCURRENT_REQUESTS_PER_ENDPOINT,
            ))
        })
        .collect();

    let model_name = model.cli_name();
    println!(
        "Evaluating model {} on {} dataset with {} samples (vLLM ports: {:?})",
        model_name, dataset_name, num_samples, vllm_ports
    );
    for endpoint in &llm_endpoints {
        println!(
            "Configured endpoint {} on port {} with max {} concurrent requests",
            endpoint.id, endpoint.vllm_port, MAX_CONCURRENT_REQUESTS_PER_ENDPOINT
        );
    }
    assert!(
        std::env::var("PYTHONPATH").is_ok(),
        "PYTHONPATH environment variable is not set"
    );
    Python::initialize();

    let client = Client::new();
    let mut rng = StdRng::seed_from_u64(42);

    let rollout_log_path = get_rollout_log_path(model, &dataset_name, num_samples);
    let rollout_log_store_for_loading = SqliteSessionLogStore::new(&rollout_log_path).unwrap();
    let asset_file_trees = AssetFileTrees {
        model,
        dataset: dataset_name.clone(),
        num_samples,
    };
    let asset_file_dataset = AssetFileDataset {
        dataset: dataset_name.clone(),
        num_samples,
    };
    let dataset_hash = asset_file_dataset.synchronize();
    let dataset = asset_file_dataset.fetch();
    let trees_tracking_file_path = asset_file_trees.version_tracking_path();
    if !std::path::Path::new(&trees_tracking_file_path).exists() {
        let tracking_content = AssetFileTreesTracking {
            dataset_hash: dataset_hash.clone(),
        };
        write_json(trees_tracking_file_path, &tracking_content).unwrap();
    }
    let trees_store = CompletedTreeStore::new(asset_file_trees.file_path()).unwrap();

    let mut tree_completed_ids: HashSet<usize> = HashSet::new();
    let mut tree_scan_statement = trees_store.statement().unwrap();
    let rows = tree_scan_statement.try_iter().unwrap();
    for row in rows {
        let tree = row.unwrap();
        tree_completed_ids.insert(tree.id);
    }

    for tree_id in &tree_completed_ids {
        rollout_log_store_for_loading
            .drop_question_table(*tree_id)
            .unwrap();
    }    
    
    let questions = dataset
        .into_iter()
        .map(|q| (q.id, q))
        .collect::<IndexMap<usize, DeepMathQuestion>>();

    let mut unfinished_tree_ids: HashSet<usize> = questions.keys().cloned().collect();
    unfinished_tree_ids.retain(|id| !tree_completed_ids.contains(id));

    let mut loaded_session_logs: IndexMap<usize, Vec<TreeAction>> = IndexMap::new();
    for unfinished_tree_id in &unfinished_tree_ids {
        let loaded_actions = rollout_log_store_for_loading
            .load_question_actions(*unfinished_tree_id)
            .unwrap();
        loaded_session_logs.insert(*unfinished_tree_id, loaded_actions);
    }

    SHUTDOWN.store(false, Ordering::SeqCst);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!("Ctrl+C received, shutting down...");
        SHUTDOWN.store(true, Ordering::SeqCst);
    });

    let (action_tx, mut action_rx) = tokio::sync::mpsc::unbounded_channel::<TreeAction>();
    let (trajectory_tx, mut trajectory_rx) =
        tokio::sync::mpsc::unbounded_channel::<CompletedTree>();
    let mut submit_task_rng = StdRng::seed_from_u64(rng.next_u64());
    let rollout_log_store_for_writer = SqliteSessionLogStore::new(&rollout_log_path).unwrap();
    let rollout_log_store_for_cleanup = SqliteSessionLogStore::new(&rollout_log_path).unwrap();

    let mut tree_scan_statement = trees_store.statement().unwrap();
    let rows = tree_scan_statement.try_iter().unwrap();
    for row in rows {
        let trajectory = row.unwrap();
        trajectory_tx.send(trajectory).unwrap();
    }

    let action_tx_for_submit = action_tx.clone();
    let trajectory_tx_for_submit = trajectory_tx.clone();
    let trees_store_for_writer = CompletedTreeStore::new(asset_file_trees.file_path()).unwrap();
    let submit_trajectory_task_handle = tokio::spawn({
        let client = client.clone();
        let action_tx = action_tx_for_submit;
        let trajectory_tx = trajectory_tx_for_submit;
        let llm_endpoints = llm_endpoints.clone();
        async move {
            let finished_count = Arc::new(AtomicUsize::new(0));
            let total_count = unfinished_tree_ids.len();
            for id in unfinished_tree_ids {
                if SHUTDOWN.load(Ordering::SeqCst) {
                    break;
                }
                let (llm_endpoint, question_slot_permit) =
                    choose_endpoint_for_question(&llm_endpoints).await;
                let unfinished_question = questions.get(&id).unwrap();
                let question = unfinished_question.question.clone();
                let reference_answer = unfinished_question.final_answer.clone();
                let action_tx = action_tx.clone();
                let trajectory_tx = trajectory_tx.clone();
                let client = client.clone();
                let loaded_session_log = loaded_session_logs.swap_remove(&id).unwrap_or_default();
                let mut task_rng = StdRng::seed_from_u64(submit_task_rng.next_u64());
                let finished_count = finished_count.clone();
                println!(
                    "Question {} assigned to endpoint {} (port {})",
                    id, llm_endpoint.id, llm_endpoint.vllm_port
                );
                tokio::spawn(async move {
                    let _question_slot_permit = question_slot_permit;
                    rollout(
                        id,
                        question,
                        reference_answer,
                        loaded_session_log,
                        llm_endpoint,
                        client,
                        model,
                        &mut task_rng,
                        action_tx,
                        trajectory_tx,
                    )
                    .await;
                    finished_count.fetch_add(1, Ordering::SeqCst);
                    println!(
                        "Trajectory {}/{} finished",
                        finished_count.load(Ordering::SeqCst),
                        total_count
                    );
                });
            }
            drop(action_tx);
            drop(trajectory_tx);
        }
    });

    let receive_action_handle = tokio::spawn(async move {
        while let Some(action_log_item) = action_rx.recv().await {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            let question_id = action_log_item.question_id();
            rollout_log_store_for_writer
                .append_action(question_id, &action_log_item)
                .unwrap();
        }
    });

    let receive_trajectory_handle = tokio::spawn(async move {
        let mut total_correct = 0usize;
        let mut total_judged = 0usize;
        while let Some(trajectory) = trajectory_rx.recv().await {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            trees_store_for_writer.upsert(trajectory.id, &trajectory).unwrap();
            rollout_log_store_for_cleanup
                .drop_question_table(trajectory.id)
                .unwrap();
            total_correct += trajectory.trajectory.correctness_ratio.numerator;
            total_judged += trajectory.trajectory.correctness_ratio.denominator;
            let running_accuracy = if total_judged == 0 {
                0.0
            } else {
                total_correct as f64 / total_judged as f64
            };
            println!(
                "Running overall accuracy from tree correctness_ratio: {}/{} ({:.2}%)",
                total_correct,
                total_judged,
                running_accuracy * 100.0
            );
        }
    });

    drop(action_tx);
    drop(trajectory_tx);

    join_all([
        submit_trajectory_task_handle,
        receive_action_handle,
        receive_trajectory_handle,
    ])
    .await;

    let mut overall_correct = 0usize;
    let mut overall_denominator = 0usize;
    let mut tree_scan_statement = trees_store.statement().unwrap();
    let rows = tree_scan_statement.try_iter().unwrap();
    for row in rows {
        let trajectory = row.unwrap();
        overall_correct += trajectory.trajectory.correctness_ratio.numerator;
        overall_denominator += trajectory.trajectory.correctness_ratio.denominator;
    }
    let accuracy = if overall_denominator == 0 {
        0.0
    } else {
        overall_correct as f64 / overall_denominator as f64
    };
    println!(
        "Rollout evaluation completed for model {} on {} dataset with {} samples. Overall tree correctness accuracy: {}/{} ({:.2}%)",
        model_name,
        dataset_name,
        num_samples,
        overall_correct,
        overall_denominator,
        accuracy * 100.0
    );
}
