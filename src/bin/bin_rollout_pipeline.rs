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
    call_llm::{call_llm_chat_completions, set_vllm_port},
    datasets::{DeepMathQuestion, get_question_path},
    deepmath::{
        generate_raw_answers::Model,
        judge_answers::{DeepMathCorrectness, get_accuracy_path, get_correctness_path},
    },
    multi_agent::{
        generate_rollout_answers::{
            RolloutTrajectory, get_rollout_trajectory_path, get_session_log_path,
        },
        rollout::rollout,
        session::{RolloutAction, RolloutActionLogItem},
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
}

// we want to log each action
// if a question finishes, record the trajectory, and evaluate its correctness immediately
// For loading, first load the trajectories and find all finished question indices, and then remove all logs related to these questions
// Then reconstruct part of unfinished trajectories from the log and continue the rollout
// If all trajectories finish, sort the trajectories, correctness and report the final accuracy.

const MAX_TASKS: usize = 1000;
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

async fn judge_rollout_answer_task(
    answer: RolloutTrajectory,
    client: Client,
) -> DeepMathCorrectness {
    let prompt = format!(
        // "The question is: {}. The model's answer is: {}. The correct answer is: {}. Please evaluate whether the model's answer is correct and return only 'correct' or 'incorrect'.",
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \
The model's answer is: \"{}\", and the correct answer is: \"{}\". Return only 'correct' or 'incorrect'.",
        answer.model_answer, answer.correct_answer
    );
    let evaluation = call_llm_chat_completions(client, prompt, Model::Gpt4o, false)
        .await
        .trim()
        .to_lowercase();
    println!("Evaluation for question {}: {}", answer.id, evaluation);
    let correct = match evaluation.as_str() {
        "correct" => true,
        "incorrect" => false,
        _ => {
            println!(
                "Unexpected evaluation result for question {}: {}. Treating it as incorrect.",
                answer.id, evaluation
            );
            false
        }
    };
    DeepMathCorrectness {
        id: answer.id,
        correct,
        model_answer: answer.model_answer,
        correct_answer: answer.correct_answer,
        question: answer.question,
    }
}

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
    } = Args::parse();
    assert!(vllm_port > 0, "--vllm-port must be greater than 0");
    set_vllm_port(vllm_port);
    let model_name = model.cli_name();
    println!(
        "Evaluating model {} on {} dataset with {} samples (vLLM port: {})",
        model_name, dataset_name, num_samples, vllm_port
    );

    let client = Client::new();
    let mut rng = StdRng::seed_from_u64(42);
    let verifier_probability = 1.0;
    // read log file
    let session_log_path = get_session_log_path(model, &dataset_name, num_samples);
    let mut session_log_items: Vec<RolloutActionLogItem> =
        read_json_lines(&session_log_path).unwrap_or_default();
    // read trajectory file
    let trajectory_path = get_rollout_trajectory_path(model, &dataset_name, num_samples);
    let trajectory_items: IndexMap<usize, RolloutTrajectory> =
        read_json_lines_indexed(&trajectory_path).unwrap_or_default();
    let correctness_path = get_correctness_path(model, &dataset_name, num_samples, true);
    let correctness_items: IndexMap<usize, DeepMathCorrectness> =
        read_json_lines_indexed(&correctness_path).unwrap_or_default();
    let trajectory_completed_ids: HashSet<usize> = trajectory_items.keys().cloned().collect();
    let correctness_completed_ids: HashSet<usize> = correctness_items.keys().cloned().collect();
    // delete parts in log file that is already finished and write back
    session_log_items.retain(|item| !trajectory_completed_ids.contains(&item.question_id));
    write_jsonl_file(&session_log_path, &session_log_items).unwrap();
    // construct a hashmap of loaded session logs
    let mut loaded_session_logs: IndexMap<usize, Vec<RolloutAction>> = IndexMap::new();
    for log_item in session_log_items {
        loaded_session_logs
            .entry(log_item.question_id)
            .or_insert_with(Vec::new)
            .push(log_item.action);
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
    let mut unfinished_correctness_ids: HashSet<usize> = questions.keys().cloned().collect();
    unfinished_correctness_ids.retain(|id| !correctness_completed_ids.contains(id));

    let sem = Arc::new(Semaphore::new(MAX_TASKS));
    SHUTDOWN.store(false, Ordering::SeqCst);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!("Ctrl+C received, shutting down...");
        SHUTDOWN.store(true, Ordering::SeqCst);
    });
    let (action_tx, mut action_rx) = tokio::sync::mpsc::unbounded_channel::<RolloutActionLogItem>();
    let (trajectory_tx, mut trajectory_rx) =
        tokio::sync::mpsc::unbounded_channel::<RolloutTrajectory>();
    let (correctness_input_tx, mut correctness_input_rx) =
        tokio::sync::mpsc::unbounded_channel::<RolloutTrajectory>();
    let (correctness_output_tx, mut correctness_output_rx) =
        tokio::sync::mpsc::unbounded_channel::<DeepMathCorrectness>();
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
    let mut correctness_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&correctness_path)
        .unwrap();
    for trajectory in trajectory_items.into_values() {
        trajectory_tx.send(trajectory).unwrap();
    }
    for correctness in correctness_items.into_values() {
        correctness_output_tx.send(correctness).unwrap();
    }
    let submit_trajectory_task_handle = tokio::spawn({
        let sem = sem.clone();
        let client = client.clone();
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
                        verifier_probability,
                        &mut task_rng,
                        action_tx,
                        trajectory_tx,
                    )
                    .await;
                    drop(permit);
                    finished_count.fetch_add(1, Ordering::SeqCst);
                    println!("Trajectory {}/{} finished", finished_count.load(Ordering::SeqCst), total_count);
                });
            }
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
        while let Some(trajectory) = trajectory_rx.recv().await {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            let json_line = serde_json::to_string(&trajectory).unwrap();
            writeln!(trajectory_file, "{}", json_line).unwrap();
            // handle correctness
            if unfinished_correctness_ids.contains(&trajectory.id) {
                // submit the task
                correctness_input_tx.send(trajectory).unwrap();
            }
        }
    });
    let submit_correctness_task_handle = tokio::spawn(async move {
        while let Some(trajectory) = correctness_input_rx.recv().await {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            let permit = sem.clone().acquire_owned().await.unwrap();
            let client = client.clone();
            let correctness_output_tx = correctness_output_tx.clone();
            tokio::spawn(async move {
                let correctness = judge_rollout_answer_task(trajectory, client).await;
                correctness_output_tx.send(correctness).unwrap();
                drop(permit);
            });
        }
    });
    let receive_correctness_handle = tokio::spawn(async move {
        // let correct_count = Arc::new(AtomicUsize::new(0));
        // let total_count = Arc::new(AtomicUsize::new(0));
        let mut correct_count: usize = 0;
        let mut total_count: usize = 0;
        while let Some(correctness) = correctness_output_rx.recv().await {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            let json_line = serde_json::to_string(&correctness).unwrap();
            writeln!(correctness_file, "{}", json_line).unwrap();
            if correctness.correct {
                correct_count += 1;
            }
            total_count += 1;
            let accuracy = correct_count as f64 / total_count as f64;
            println!("Running accuracy: {}/{} ({:.2}%)", correct_count, total_count, accuracy * 100.0);
        }
    });

    join_all([
        submit_trajectory_task_handle,
        receive_action_handle,
        receive_trajectory_handle,
        submit_correctness_task_handle,
        receive_correctness_handle,
    ])
    .await;
    // read trajectory file and correctness file, and sort them by id
    let mut trajectories: Vec<RolloutTrajectory> =
        read_json_lines(&trajectory_path).unwrap_or_default();
    trajectories.sort_by_key(|t| t.id);
    let mut correctness: Vec<DeepMathCorrectness> =
        read_json_lines(&correctness_path).unwrap_or_default();
    correctness.sort_by_key(|c| c.id);
    // write back sorted trajectories and correctness
    write_jsonl_file(&trajectory_path, &trajectories).unwrap();
    write_jsonl_file(&correctness_path, &correctness).unwrap();
    let accuracy_file_path = get_accuracy_path(model, &dataset_name, num_samples, true);
    let mut correct_count = 0;
    for c in &correctness {
        if c.correct {
            correct_count += 1;
        }
    }
    let accuracy = correct_count as f64 / correctness.len() as f64;
    std::fs::write(
        accuracy_file_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "total": correctness.len(),
            "correct": correct_count,
            "accuracy": accuracy,
        }))
        .unwrap(),
    )
    .unwrap();
    println!(
        "Rollout evaluation completed for model {} on {} dataset with {} samples. Accuracy: {:.2}%",
        model_name, dataset_name, num_samples, accuracy * 100.0
    );
}
