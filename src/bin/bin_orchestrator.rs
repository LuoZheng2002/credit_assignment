use std::{backtrace::Backtrace, sync::Arc};

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::{read_json, read_toml},
    llm_model::{Gpt4o, LlmModelName, Qwen3_4B, Qwen25_7B, Qwen35_4B, Qwen35_08B},
    orchestrator::{OrchestrationProgress, Orchestrator},
    python_training_config::PythonTrainingConfigCommon,
};
use research_utility::{log_message::log_info, progress_screen::ProgressScreen};
use tokio::sync::Semaphore;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run direct tree rollout and save action logs"
)]
struct Args {
    #[arg(long)]
    model_cli_name: String,
    #[arg(long)]
    max_rollout_concurrency: usize,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    validation_rollout_config_path: String,
    #[arg(long)]
    training_set_rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    num_total_epochs: usize,
    #[arg(long)]
    max_num_training_trajectories: usize,
    #[arg(long)]
    training_config_common_path: String,
    #[arg(long)]
    first_n_training_samples: Option<usize>,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long)]
    first_n_rollout_samples: Option<usize>,
    #[arg(long, default_value_t = 1)]
    max_sqlite_connections: u32,
    #[arg(long)]
    sglang_server_log_path: Option<String>,
    #[arg(long)]
    message_log_path: Option<String>,
    #[arg(long)]
    progress_save_file_path: String,
    #[arg(long)]
    num_gpus: usize,
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
        model_cli_name,
        config_nickname,
        max_rollout_concurrency,
        validation_rollout_config_path,
        training_set_rollout_config_path,
        posterior_hyperparameters_path,
        num_total_epochs,
        ui,
        first_n_rollout_samples,
        max_sqlite_connections,
        sglang_server_log_path,
        message_log_path,
        progress_save_file_path,
        max_num_training_trajectories,
        training_config_common_path,
        first_n_training_samples,
        num_gpus,
    } = Args::parse();
    // maybe we need to initialize python here

    if ui {
        ProgressScreen::initialize("Orchestrator Progress", true, message_log_path)
            .await
            .unwrap();
    }
    let validation_rollout_config: DirectRolloutConfig =
        read_json(validation_rollout_config_path).unwrap();
    let training_set_rollout_config: DirectRolloutConfig =
        read_json(training_set_rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let training_config_common: PythonTrainingConfigCommon =
        read_toml(training_config_common_path).unwrap();
    // do the rest of the orchestrator work here
    let client = reqwest::Client::new();
    let question_semaphore = Arc::new(Semaphore::new(max_rollout_concurrency));
    let progress = match read_json::<OrchestrationProgress>(&progress_save_file_path) {
        Ok(progress) => {
            log_info(format!(
                "Loaded progress from file: {}. Progress: {:?}",
                progress_save_file_path, progress
            ));
            progress
        }
        Err(_) => {
            log_info(format!(
                "No progress file found at: {}, starting fresh",
                progress_save_file_path
            ));
            OrchestrationProgress::WorkingOnValidation { epoch: 0 }
        }
    };
    let mut orchestrator = Orchestrator {
        config_nickname,
        validation_rollout_config,
        training_set_rollout_config,
        posterior_calculation_config,
        num_total_epochs,
        // max_rollout_concurrency,
        first_n_rollout_samples,
        max_sqlite_connections,
        client,
        question_semaphore: question_semaphore,
        inference_server_handle: None,
        sglang_server_log_path,
        max_num_training_trajectories,
        training_config_common,
        first_n_training_samples,
        progress_save_file_path,
        progress,
        num_gpus,
    };
    let model_name = LlmModelName::from_str(&model_cli_name, true).unwrap();
    match model_name {
        LlmModelName::Gpt4o => orchestrator.orchestrate::<Gpt4o>().await,
        LlmModelName::Qwen3_4b => orchestrator.orchestrate::<Qwen3_4B>().await,
        LlmModelName::Qwen25_7b => orchestrator.orchestrate::<Qwen25_7B>().await,
        LlmModelName::Qwen35_08b => orchestrator.orchestrate::<Qwen35_08B>().await,
        LlmModelName::Qwen35_4b => orchestrator.orchestrate::<Qwen35_4B>().await,
    }
    if ui {
        ProgressScreen::shutdown().await.unwrap();
    }
}
