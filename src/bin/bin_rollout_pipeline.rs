use std::{
    collections::HashSet,
    fs::OpenOptions,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use clap::Parser;
use credit_assignment::{
    call_llm::set_vllm_port,
    datasets::{DeepMathQuestion, get_question_path},
    deepmath::generate_raw_answers::Model,
    multi_agent::{
        generate_rollout_answers::{
            RolloutTree, get_rollout_trajectory_path, get_session_log_path,
        },
        rollout::rollout,
        session::TreeUpdateEvent,
    },
    parallel_process_jsonl::{read_json_lines, read_json_lines_indexed, write_jsonl_file},
};
use futures::future::join_all;
use indexmap::IndexMap;
use pyo3::Python;
use rand::{Rng, SeedableRng, rngs::StdRng};
use reqwest::Client;
use std::io::Write;
use tokio::sync::Semaphore;

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: Model,
    #[arg(long)]
    vllm_port: u16,
    #[arg(long, default_value_t = 1000)]
    max_tasks: usize,
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
        max_tasks,
        take_over_mode_decision,
    } = Args::parse();
    assert!(vllm_port > 0, "--vllm-port must be greater than 0");
    set_vllm_port(vllm_port);
    let model_name = model.cli_name();
    println!(
        "Evaluating model {} on {} dataset with {} samples (vLLM port: {})",
        model_name, dataset_name, num_samples, vllm_port
    );
    println!(
        "max_tasks: {}, take_over_mode_decision: {}",
        max_tasks, take_over_mode_decision
    );
    let client = Client::new();
    let mut rng = StdRng::seed_from_u64(42);
    // read log file
    let session_log_path = get_session_log_path(model, &dataset_name, num_samples);
    let mut session_log_items: Vec<TreeUpdateEvent> =
        read_json_lines(&session_log_path).unwrap_or_default();
    // read trajectory file
    let trajectory_path = get_rollout_trajectory_path(model, &dataset_name, num_samples);
    let trajectory_items: IndexMap<usize, RolloutTree> =
        read_json_lines_indexed(&trajectory_path).unwrap_or_default();
    let trajectory_completed_ids: HashSet<usize> = trajectory_items.keys().cloned().collect();
    // delete parts in log file that is already finished and write back
    session_log_items.retain(|item| !trajectory_completed_ids.contains(&item.question_id()));
    write_jsonl_file(&session_log_path, &session_log_items).unwrap();
    // construct a hashmap of loaded session logs
    let mut loaded_session_logs: IndexMap<usize, Vec<TreeUpdateEvent>> = IndexMap::new();
    for log_item in session_log_items {
        loaded_session_logs
            .entry(log_item.question_id())
            .or_insert_with(Vec::new)
            .push(log_item);
    }
    // load questions and answers for unfinished trajectories
    let question_path = get_question_path(&dataset_name, num_samples);
    let questions: IndexMap<usize, DeepMathQuestion> =
        read_json_lines_indexed(&question_path).unwrap();
    // let mut unfinished_trajectories: IndexMap<usize, DeepMathQuestion> =
    //     read_json_lines_indexed(&question_path).unwrap();
    // unfinished_trajectories.retain(|id, _| !trajectory_completed_ids.contains(id));
    let mut unfinished_trajectory_ids: HashSet<usize> = questions.keys().cloned().collect();
    unfinished_trajectory_ids.retain(|id| !trajectory_completed_ids.contains(id));

    let sem = Arc::new(Semaphore::new(max_tasks));
    SHUTDOWN.store(false, Ordering::SeqCst);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!("Ctrl+C received, shutting down...");
        SHUTDOWN.store(true, Ordering::SeqCst);
    });
    let (action_tx, mut action_rx) = tokio::sync::mpsc::unbounded_channel::<TreeUpdateEvent>();
    let (trajectory_tx, mut trajectory_rx) =
        tokio::sync::mpsc::unbounded_channel::<RolloutTree>();
    let mut sumit_task_rng = StdRng::seed_from_u64(rng.next_u64());
    let mut session_log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&session_log_path)
        .unwrap();
    let mut trajectory_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // truncate and rewrite history to let the items flow through the channel
        .open(&trajectory_path)
        .unwrap();
    for trajectory in trajectory_items.into_values() {
        trajectory_tx.send(trajectory).unwrap();
    }
    let action_tx_for_submit = action_tx.clone();
    let trajectory_tx_for_submit = trajectory_tx.clone();
    let submit_trajectory_task_handle = tokio::spawn({
        let sem = sem.clone();
        let client = client.clone();
        let action_tx = action_tx_for_submit;
        let trajectory_tx = trajectory_tx_for_submit;
        async move {
            let finished_count = Arc::new(AtomicUsize::new(0));
            let total_count = unfinished_trajectory_ids.len();
            for id in unfinished_trajectory_ids {
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
            let json_line = serde_json::to_string(&action_log_item).unwrap();
            writeln!(session_log_file, "{}", json_line).unwrap();
        }
    });
    let receive_trajectory_handle = tokio::spawn(async move {
        let mut total_correct = 0usize;
        let mut total_judged = 0usize;
        while let Some(trajectory) = trajectory_rx.recv().await {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            let json_line = serde_json::to_string(&trajectory).unwrap();
            writeln!(trajectory_file, "{}", json_line).unwrap();
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
    // read trajectory file and sort by id
    let mut trajectories: Vec<RolloutTree> =
        read_json_lines(&trajectory_path).unwrap_or_default();
    trajectories.sort_by_key(|t| t.id);
    // write back sorted trajectories
    write_jsonl_file(&trajectory_path, &trajectories).unwrap();

    let mut overall_correct = 0usize;
    let mut overall_denominator = 0usize;
    for trajectory in &trajectories {
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
