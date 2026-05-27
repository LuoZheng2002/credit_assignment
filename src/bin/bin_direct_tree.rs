use std::{backtrace::Backtrace, sync::Arc};

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        direct_rollout::direct_rollout_all_with_config,
        direct_rollout_config::DirectRolloutConfig,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{Gpt4o, LlmCliArgs, LlmModelName, Qwen3_4B, Qwen25, Qwen35_4B, Qwen35_08B},
};
use pyo3::Python;
use reqwest::Client;
use research_utility::{
    message::WorkerMessage,
    progress_screen::{ProgressScreen, ProgressScreenConfig},
    worker_message_tx::WORKER_MESSAGE_TX,
};
use tokio::sync::Semaphore;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run direct tree rollout and save action logs"
)]
struct Args {
    #[command(flatten)]
    llm_cli_args: LlmCliArgs,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long)]
    first_n_samples: Option<usize>,
}

#[tokio::main]
async fn main() {
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
        config_nickname,
        llm_cli_args,
        rollout_config_path,
        posterior_hyperparameters_path,
        ui,
        first_n_samples,
    } = Args::parse();
    Python::initialize();
    check_sympy_availability().unwrap();

    println!("Starting direct rollout evaluation pipeline...");
    let client = Client::new();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    if rollout_config.accuracy_under_temperature.is_none() {
        eprintln!(
            "WARNING: rollout_config.accuracy_under_temperature is None; all segment posteriors will use mean=0 and std=1."
        );
    }
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let question_semaphore = Arc::new(Semaphore::new(200));
    let model_name = LlmModelName::from_str(&llm_cli_args.model_cli_name, true).unwrap();
    // set up ui
    let progress_screen = if ui {
        let mut progress_screen_config = ProgressScreenConfig::from_defaults(1, 1);
        progress_screen_config.window_title = "Bin Direct Tree Rollout Progress".to_string();
        progress_screen_config.persist_after_channel_close = false;

        let progress_screen = ProgressScreen::new(progress_screen_config);
        Some(progress_screen)
    } else {
        None
    };
    let (worker_message_tx, mut worker_message_rx) = tokio::sync::mpsc::unbounded_channel();
    WORKER_MESSAGE_TX.store(Some(Arc::new(worker_message_tx)));
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
    match model_name {
        LlmModelName::Qwen25_7b => {
            direct_rollout_all_with_config::<Qwen25>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                client,
                question_semaphore,
                &llm_cli_args,
                first_n_samples,
            )
            .await;
        }
        LlmModelName::Qwen3_4b => {
            direct_rollout_all_with_config::<Qwen3_4B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                client,
                question_semaphore,
                &llm_cli_args,
                first_n_samples,
            )
            .await;
        }
        LlmModelName::Qwen35_4b => {
            direct_rollout_all_with_config::<Qwen35_4B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                client,
                question_semaphore,
                &llm_cli_args,
                first_n_samples,
            )
            .await;
        }
        LlmModelName::Qwen35_08b => {
            direct_rollout_all_with_config::<Qwen35_08B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                client,
                question_semaphore,
                &llm_cli_args,
                first_n_samples,
            )
            .await;
        }
        LlmModelName::Gpt4o => {
            direct_rollout_all_with_config::<Gpt4o>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
                client,
                question_semaphore,
                &llm_cli_args,
                first_n_samples,
            )
            .await;
        }
    }
    println!("All rollouts completed, shutting down worker message listener...");
    WORKER_MESSAGE_TX.store(None);
    worker_message_listener.await.unwrap();
}
