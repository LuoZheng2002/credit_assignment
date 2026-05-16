use std::{
    backtrace::Backtrace,
    sync::Arc,
};

use clap::{ArgAction, Parser};
use credit_assignment::{
    agent::rollout_batch::rollout_batch,
    llm_model::LlmModel,
    message::WorkerMessage,
    progress_screen::ProgressScreenConfig,
    progress_screen::ProgressScreen,
    worker_message_tx::WORKER_MESSAGE_TX,
};
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: LlmModel,
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    vllm_ports: Vec<u16>,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
}

// we want to log each action
// if a question finishes, record the trajectory immediately
// For loading, first load trajectories and find finished question indices, then remove all logs related to these questions
// Then reconstruct unfinished trajectories from logs and continue the rollout
// If all trajectories finish, sort trajectories and report final overall tree correctness accuracy.

#[tokio::main]
async fn main() {
    println!("Starting rollout evaluation pipeline...");
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        let rust_backtrace = std::env::var("RUST_BACKTRACE").ok();
        if matches!(rust_backtrace.as_deref(), Some("1") | Some("full")) {
            let backtrace = Backtrace::force_capture();
            eprintln!("backtrace:\n{}", backtrace);
        }
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    let Args {
        dataset_name,
        model,
        num_samples,
        vllm_ports,
        ui,
    } = Args::parse();

    let (worker_message_tx, mut worker_message_rx) = mpsc::unbounded_channel();
    WORKER_MESSAGE_TX.store(Some(Arc::new(worker_message_tx)));

    let progress_screen = if ui {
        let mut progress_screen_config = ProgressScreenConfig::from_defaults(vllm_ports.len(), 1);
        progress_screen_config.window_title = "Bin Tree Rollout Progress".to_string();
        progress_screen_config.key_order = vec![
            "status".to_string(),
            "model".to_string(),
            "dataset".to_string(),
            "num_samples".to_string(),
            "endpoints".to_string(),
            "running_accuracy".to_string(),
        ];
        progress_screen_config.persist_after_channel_close = false;

        let progress_screen = ProgressScreen::new(progress_screen_config);
        Some(progress_screen)
    } else {
        None
    };

    let worker_message_listener = tokio::spawn(async move {
        while let Some(message) = worker_message_rx.recv().await {
            if let Some(progress_screen) = &progress_screen {
                progress_screen.receive_message(message);
            } else {
                match message {
                    WorkerMessage::KeyValuePair { key, value } => {
                        println!("{key}: {value}");
                    }
                    WorkerMessage::WorkerProgress {
                        worker_id,
                        progress,
                        label,
                    } => {
                        println!("worker {worker_id} progress {progress:.3}: {label}");
                    }
                    WorkerMessage::MasterProgress { progress, label } => {
                        println!("master progress {progress:.3}: {label}");
                    }
                }
            }
        }
    });

    rollout_batch(model, dataset_name, num_samples, vllm_ports).await;

    WORKER_MESSAGE_TX.store(None);
    worker_message_listener
        .await
        .expect("worker message listener should complete without panic");
}
