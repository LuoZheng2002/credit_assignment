use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use clap::Parser;
use credit_assignment::{
    agent::{
        rollout_loop::rollout,
        tree_action::TreeAction,
        tree_schema::{
            AssetFileTrees, AssetFileTreesTracking, CompletedTree, CompletedTreeStore,
            get_rollout_log_path,
        },
    },
    call_llm::set_vllm_port,
    datasets::{AssetFileDataset, DeepMathQuestion},
    direct_answer::generate_raw_answers::LlmModel,
    parallel_process_jsonl::write_json,
    sqlite_session_log::SqliteSessionLogStore,
    version_tracking::AssetFile,
};
use futures::future::join_all;
use indexmap::IndexMap;
use pyo3::Python;
use rand::{Rng, SeedableRng, rngs::StdRng};
use reqwest::Client;
use tokio::sync::Semaphore;

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: LlmModel,
    #[arg(long)]
    vllm_port: u16,
    #[arg(long, action = clap::ArgAction::Set)]
    take_over_mode_decision: bool,
}

// we want to log each action
// if a question finishes, record the trajectory immediately
// For loading, first load trajectories and find finished question indices, then remove all logs related to these questions
// Then reconstruct unfinished trajectories from logs and continue the rollout
// If all trajectories finish, sort trajectories and report final overall tree correctness accuracy.

// const MAX_TASKS: usize = 100;
static SHUTDOWN: AtomicBool = AtomicBool::new(false);
const MAX_CONCURRENT_ROLLOUT_JOBS: usize = 10;

#[tokio::main]
async fn main() {
    println!("Starting rollout evaluation pipeline...");
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    assert!(
        std::env::var("PYTHONPATH").is_ok(),
        "PYTHONPATH environment variable is not set"
    );
    Python::initialize();
    let Args {
        dataset_name,
        model,
        num_samples,
        vllm_port,
        take_over_mode_decision,
    } = Args::parse();
    assert!(vllm_port > 0, "--vllm-port must be greater than 0");
    set_vllm_port(vllm_port);
    let model_name = model.cli_name();
    println!(
        "Evaluating model {} on {} dataset with {} samples (vLLM port: {})",
        model_name, dataset_name, num_samples, vllm_port
    );
    println!("take_over_mode_decision: {}", take_over_mode_decision);
    let client = Client::new();
    let mut rng = StdRng::seed_from_u64(42);
    // read log file
    let rollout_log_path = get_rollout_log_path(model, &dataset_name, num_samples);
    let rollout_log_store_for_loading = SqliteSessionLogStore::new(&rollout_log_path).unwrap();
    // read trajectory file
    // let trajectory_path = get_rollout_trees_path(model, &dataset_name, num_samples);
    let asset_file_trees = AssetFileTrees {
        model,
        dataset: dataset_name.clone(),
        num_samples,
    };
    let trees_store = asset_file_trees.fetch();
    let mut tree_completed_ids: HashSet<usize> = HashSet::new();
    let mut tree_scan_statement = trees_store.statement().unwrap();
    let rows = tree_scan_statement.try_iter().unwrap();
    for row in rows {
        let tree = row.unwrap();
        tree_completed_ids.insert(tree.id);
    }
    // delete tables in session log database that are already finished
    for tree_id in &tree_completed_ids {
        rollout_log_store_for_loading
            .drop_question_table(*tree_id)
            .unwrap();
    }
    // load questions and answers for unfinished trajectories
    let asset_file_dataset = AssetFileDataset {
        dataset: dataset_name.clone(),
        num_samples,
    };
    let dataset_hash = asset_file_dataset.synchronize();
    let dataset = asset_file_dataset.fetch();
    let questions = dataset
        .into_iter()
        .map(|q| (q.id, q))
        .collect::<IndexMap<usize, DeepMathQuestion>>();

    let mut unfinished_tree_ids: HashSet<usize> = questions.keys().cloned().collect();
    unfinished_tree_ids.retain(|id| !tree_completed_ids.contains(id));
    // construct a hashmap of loaded session logs
    let mut loaded_session_logs: IndexMap<usize, Vec<TreeAction>> = IndexMap::new();
    for unfinished_tree_id in &unfinished_tree_ids {
        let loaded_actions = rollout_log_store_for_loading
            .load_question_actions(*unfinished_tree_id)
            .unwrap();
        loaded_session_logs.insert(*unfinished_tree_id, loaded_actions);
    }

    let rollout_concurrency = MAX_CONCURRENT_ROLLOUT_JOBS;
    println!("rollout concurrency limit: {}", rollout_concurrency);
    let sem = Arc::new(Semaphore::new(rollout_concurrency));
    SHUTDOWN.store(false, Ordering::SeqCst);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!("Ctrl+C received, shutting down...");
        SHUTDOWN.store(true, Ordering::SeqCst);
    });
    let (action_tx, mut action_rx) = tokio::sync::mpsc::unbounded_channel::<TreeAction>();
    let (trajectory_tx, mut trajectory_rx) =
        tokio::sync::mpsc::unbounded_channel::<CompletedTree>();
    let mut sumit_task_rng = StdRng::seed_from_u64(rng.next_u64());
    let rollout_log_store_for_writer = SqliteSessionLogStore::new(&rollout_log_path).unwrap();
    let rollout_log_store_for_cleanup = SqliteSessionLogStore::new(&rollout_log_path).unwrap();
    // write the tracking file directly to mark the existence of the tree file.
    let tracking_content = AssetFileTreesTracking { dataset_hash };
    write_json(asset_file_trees.version_tracking_path(), &tracking_content).unwrap();
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
        let sem = sem.clone();
        let client = client.clone();
        let action_tx = action_tx_for_submit;
        let trajectory_tx = trajectory_tx_for_submit;
        async move {
            let finished_count = Arc::new(AtomicUsize::new(0));
            let total_count = unfinished_tree_ids.len();
            for id in unfinished_tree_ids {
                if SHUTDOWN.load(Ordering::SeqCst) {
                    break;
                }
                let permit = sem.clone().acquire_owned().await.unwrap();
                let unfinished_question = questions.get(&id).unwrap();
                let question = unfinished_question.question.clone();
                let reference_answer = unfinished_question.final_answer.clone();
                let action_tx = action_tx.clone();
                let trajectory_tx = trajectory_tx.clone();
                let client = client.clone();
                let loaded_session_log = loaded_session_logs.swap_remove(&id).unwrap_or_default();
                let mut task_rng = StdRng::seed_from_u64(sumit_task_rng.next_u64());
                let finished_count = finished_count.clone();
                tokio::spawn(async move {
                    rollout(
                        id,
                        question,
                        reference_answer,
                        loaded_session_log,
                        client,
                        model,
                        take_over_mode_decision,
                        &mut task_rng,
                        action_tx,
                        trajectory_tx,
                    )
                    .await;
                    drop(permit);
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
            trees_store_for_writer
                .upsert(trajectory.id, &trajectory)
                .unwrap();
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
