use std::{backtrace::Backtrace, sync::Arc};

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    agent::rollout_batch::rollout_batch,
    llm_model::LlmModel,
    llm_models::{
        Gpt4o, Gpt5Mini, LlmCliArgs, LlmModelMarker, Qwen3_4B, Qwen3_8B, Qwen25, Qwen35_4B,
    },
    message::WorkerMessage,
    progress_screen::ProgressScreen, progress_screen::ProgressScreenConfig,
    worker_message_tx::WORKER_MESSAGE_TX,
};
use reqwest::Client;
use tokio::sync::mpsc;

async fn run_rollout_for_marker<M: LlmModelMarker>(
    model: LlmModel,
    dataset_name: String,
    num_samples: usize,
    llm_cli_args: &LlmCliArgs,
) {
    let llm_callable = M::callable_from_cli_args(Client::new(), llm_cli_args);
    rollout_batch::<M, M::Callable>(model, dataset_name, num_samples, llm_callable).await;
}

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,
    #[arg(short, long)]
    num_samples: usize,
    #[command(flatten)]
    llm_cli_args: LlmCliArgs,
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
        num_samples,
        llm_cli_args,
        ui,
    } = Args::parse();

    let model = LlmModel::from_str(&llm_cli_args.model_cli_name, false)
        .expect("--model-cli-name must match one of the supported model CLI names");
    let (worker_message_tx, mut worker_message_rx) = mpsc::unbounded_channel();
    WORKER_MESSAGE_TX.store(Some(Arc::new(worker_message_tx)));

    let progress_screen = if ui {
        let mut progress_screen_config = ProgressScreenConfig::from_defaults(1, 1);
        progress_screen_config.window_title = "Bin Tree Rollout Progress".to_string();
        progress_screen_config.key_order = vec![
            "status".to_string(),
            "model".to_string(),
            "dataset".to_string(),
            "num_samples".to_string(),
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

    match model {
        LlmModel::Gpt4o => {
            run_rollout_for_marker::<Gpt4o>(model, dataset_name, num_samples, &llm_cli_args).await
        }
        LlmModel::Gpt5Mini => {
            run_rollout_for_marker::<Gpt5Mini>(model, dataset_name, num_samples, &llm_cli_args)
                .await
        }
        LlmModel::Qwen25_7b => {
            run_rollout_for_marker::<Qwen25>(model, dataset_name, num_samples, &llm_cli_args)
                .await
        }
        LlmModel::Qwen3_4b => {
            run_rollout_for_marker::<Qwen3_4B>(model, dataset_name, num_samples, &llm_cli_args)
                .await
        }
        LlmModel::Qwen3_8b => {
            run_rollout_for_marker::<Qwen3_8B>(model, dataset_name, num_samples, &llm_cli_args)
                .await
        }
        LlmModel::Qwen35_4b => {
            run_rollout_for_marker::<Qwen35_4B>(model, dataset_name, num_samples, &llm_cli_args)
                .await
        }
    }

    WORKER_MESSAGE_TX.store(None);
    worker_message_listener
        .await
        .expect("worker message listener should complete without panic");
}
