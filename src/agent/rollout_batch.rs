use std::{
    collections::{HashSet, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use futures::stream::{FuturesUnordered, StreamExt};
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
    asset_file::AssetFile,
    worker_message_tx::{log_key_value_pair, log_master_progress, log_worker_progress},
};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
const MAX_CONCURRENT_REQUESTS_PER_ENDPOINT: usize = 100;

pub enum LogOrTree {
    Action(TreeAction),
    Tree(CompletedTree),
}

async fn choose_endpoint_for_question(
    llm_endpoints: &[Arc<LlmEndpoint>],
) -> (Arc<LlmEndpoint>, OwnedSemaphorePermit) {
    assert!(!llm_endpoints.is_empty(), "llm_endpoints must not be empty");

    let (most_available_endpoint_idx, max_available_slots) = llm_endpoints
        .iter()
        .enumerate()
        .max_by_key(|(_, endpoint)| endpoint.question_slot_semaphore.available_permits())
        .expect("llm_endpoints must not be empty");

    if max_available_slots
        .question_slot_semaphore
        .available_permits()
        > 0
    {
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
    log_key_value_pair(
        "status".to_string(),
        "Initializing rollout batch".to_string(),
    );
    log_key_value_pair("model".to_string(), model_name.to_string());
    log_key_value_pair("dataset".to_string(), dataset_name.clone());
    log_key_value_pair("num_samples".to_string(), num_samples.to_string());
    log_key_value_pair("endpoints".to_string(), format!("{}", vllm_ports.len()));

    for endpoint in &llm_endpoints {
        log_key_value_pair(
            "status".to_string(),
            format!(
                "Configured endpoint {} on port {} with max {} concurrent requests",
                endpoint.id, endpoint.vllm_port, MAX_CONCURRENT_REQUESTS_PER_ENDPOINT
            ),
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
    log_key_value_pair("status".to_string(), "Dataset loaded".to_string());
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
            .drop_table(*tree_id)
            .unwrap();
    }

    let questions = dataset
        .into_iter()
        .map(|q| (q.id, q))
        .collect::<IndexMap<usize, DeepMathQuestion>>();
    let total_questions = questions.len();

    let mut unfinished_tree_ids: HashSet<usize> = questions.keys().cloned().collect();
    unfinished_tree_ids.retain(|id| !tree_completed_ids.contains(id));

    let mut loaded_session_logs: IndexMap<usize, Vec<TreeAction>> = IndexMap::new();
    for unfinished_tree_id in &unfinished_tree_ids {
        let loaded_actions = rollout_log_store_for_loading
            .load_table(*unfinished_tree_id)
            .unwrap();
        loaded_session_logs.insert(*unfinished_tree_id, loaded_actions);
    }
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

    let (log_or_tree_tx, mut log_or_tree_rx) = tokio::sync::mpsc::unbounded_channel::<LogOrTree>();
    let mut submit_task_rng = StdRng::seed_from_u64(rng.next_u64());
    let rollout_log_store_for_writer = SqliteSessionLogStore::new(&rollout_log_path).unwrap();
    let processed_question_count = Arc::new(AtomicUsize::new(tree_completed_ids.len()));
    publish_master_progress(tree_completed_ids.len(), total_questions);

    let endpoint_stats_shutdown = Arc::new(AtomicBool::new(false));
    let endpoint_stats_handle = tokio::spawn({
        let llm_endpoints = llm_endpoints.clone();
        let endpoint_stats_shutdown = endpoint_stats_shutdown.clone();
        async move {
            let mut snapshots: VecDeque<(Instant, Vec<usize>)> = VecDeque::new();
            let mut redraw_interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                redraw_interval.tick().await;
                if SHUTDOWN.load(Ordering::SeqCst) || endpoint_stats_shutdown.load(Ordering::SeqCst)
                {
                    break;
                }

                let now = Instant::now();
                let completed_counts: Vec<usize> = llm_endpoints
                    .iter()
                    .map(|endpoint| endpoint.completed_requests.load(Ordering::SeqCst))
                    .collect();
                snapshots.push_back((now, completed_counts.clone()));
                while snapshots.len() > 1
                    && now.duration_since(
                        snapshots
                            .front()
                            .expect("snapshot queue must be non-empty")
                            .0,
                    ) > Duration::from_secs(5)
                {
                    snapshots.pop_front();
                }

                let (window_start_time, window_start_counts) =
                    snapshots.front().expect("snapshot queue must be non-empty");
                let elapsed_seconds = now
                    .duration_since(*window_start_time)
                    .as_secs_f64()
                    .max(1e-6);

                for endpoint in &llm_endpoints {
                    let completed = endpoint.completed_requests.load(Ordering::SeqCst);
                    let baseline_completed = window_start_counts
                        .get(endpoint.id)
                        .copied()
                        .expect("window_start_counts must include every endpoint");
                    let req_per_second =
                        completed.saturating_sub(baseline_completed) as f64 / elapsed_seconds;

                    let pending_requests = endpoint
                        .max_concurrent_requests
                        .saturating_sub(endpoint.request_semaphore.available_permits());
                    let pending_ratio =
                        pending_requests as f32 / endpoint.max_concurrent_requests as f32;
                    log_worker_progress(
                        endpoint.id,
                        pending_ratio,
                        format!(
                            ":{} | {:.2} req/s (5s) | pending {}/{}",
                            endpoint.vllm_port,
                            req_per_second,
                            pending_requests,
                            endpoint.max_concurrent_requests,
                        ),
                    );
                }
            }
        }
    });

    let mut tree_scan_statement = trees_store.statement().unwrap();
    let rows = tree_scan_statement.try_iter().unwrap();
    for row in rows {
        let trajectory = row.unwrap();
        log_or_tree_tx.send(LogOrTree::Tree(trajectory)).unwrap();
    }

    let log_or_tree_tx_for_submit = log_or_tree_tx.clone();
    let trees_store_for_writer = CompletedTreeStore::new(asset_file_trees.file_path()).unwrap();
    let submit_trajectory_task_handle = tokio::spawn({
        let client = client.clone();
        let log_or_tree_tx = log_or_tree_tx_for_submit;
        let llm_endpoints = llm_endpoints.clone();
        async move {
            let finished_count = Arc::new(AtomicUsize::new(0));
            // let total_count = unfinished_tree_ids.len();
            for id in unfinished_tree_ids {
                if SHUTDOWN.load(Ordering::SeqCst) {
                    break;
                }
                let (llm_endpoint, question_slot_permit) =
                    choose_endpoint_for_question(&llm_endpoints).await;
                let unfinished_question = questions.get(&id).unwrap();
                let question = unfinished_question.question.clone();
                let reference_answer = unfinished_question.final_answer.clone();
                let log_or_tree_tx = log_or_tree_tx.clone();
                let client = client.clone();
                let loaded_session_log = loaded_session_logs.swap_remove(&id).unwrap_or_default();
                let mut task_rng = StdRng::seed_from_u64(submit_task_rng.next_u64());
                let finished_count = finished_count.clone();
                log_key_value_pair(
                    "status".to_string(),
                    format!(
                        "Question {} assigned to endpoint {} (port {})",
                        id, llm_endpoint.id, llm_endpoint.vllm_port
                    ),
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
                        log_or_tree_tx,
                    )
                    .await;
                    finished_count.fetch_add(1, Ordering::SeqCst);
                });
            }
            drop(log_or_tree_tx);
        }
    });

    let receive_log_or_tree_handle = tokio::spawn(async move {
        let mut total_correct = 0usize;
        let mut total_judged = 0usize;
        let existing_completed_ids = tree_completed_ids;
        let processed_question_count = processed_question_count;
        while let Some(log_or_tree_item) = log_or_tree_rx.recv().await {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            match log_or_tree_item {
                LogOrTree::Action(action_log_item) => {
                    let question_id = action_log_item.question_id();
                    rollout_log_store_for_writer
                        .append(question_id, &action_log_item)
                        .unwrap();
                }
                LogOrTree::Tree(trajectory) => {
                    trees_store_for_writer
                        .upsert(trajectory.id, &trajectory)
                        .unwrap();
                    rollout_log_store_for_writer
                        .drop_table(trajectory.id)
                        .unwrap();
                    total_correct += trajectory.trajectory.correctness_ratio.numerator;
                    total_judged += trajectory.trajectory.correctness_ratio.denominator;
                    let running_accuracy = if total_judged == 0 {
                        0.0
                    } else {
                        total_correct as f64 / total_judged as f64
                    };
                    log_key_value_pair(
                        "running_accuracy".to_string(),
                        format!(
                            "{}/{} ({:.2}%)",
                            total_correct,
                            total_judged,
                            running_accuracy * 100.0
                        ),
                    );

                    if !existing_completed_ids.contains(&trajectory.id) {
                        let processed = processed_question_count.fetch_add(1, Ordering::SeqCst) + 1;
                        publish_master_progress(processed, total_questions);
                    }
                }
            }
        }
    });

    drop(log_or_tree_tx);

    submit_trajectory_task_handle.await.unwrap();
    receive_log_or_tree_handle.await.unwrap();

    endpoint_stats_shutdown.store(true, Ordering::SeqCst);
    endpoint_stats_handle.await.unwrap();
    log_key_value_pair("status".to_string(), "Finalizing summary".to_string());

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
    log_key_value_pair(
        "status".to_string(),
        format!(
            "Rollout evaluation completed. Overall tree correctness accuracy: {}/{} ({:.2}%)",
            overall_correct,
            overall_denominator,
            accuracy * 100.0
        ),
    );
    publish_master_progress(total_questions, total_questions);
    log_key_value_pair("status".to_string(), "Completed".to_string());
}
