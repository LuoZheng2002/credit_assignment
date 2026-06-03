use std::backtrace::Backtrace;

use clap::{ArgAction, Parser, ValueEnum};
use std::collections::BTreeMap;

use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        direct_rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::{read_json, read_toml},
    llm_model::{Gpt4o, LlmModelName, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B},
    orchestrator::{OrchestrationProgress, OrchestrationStatus, Orchestrator},
    python_training_config::PythonTrainingConfigCommon,
};
use research_utility::progress_tui_server::{
    ProgressTuiServer, log_exit_hint, log_info, log_warning,
};

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
    training_rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    num_total_epochs: usize,
    #[arg(long)]
    cumulative_avg_abs_advantage_cutoff: f32,
    #[arg(long, value_enum)]
    advantage_calculation_policy: AdvantageCalculationPolicy,
    #[arg(long)]
    training_config_common_path: String,
    #[arg(long)]
    training_time: f32,
    #[arg(long)]
    num_iterations_limit: usize,
    #[arg(long)]
    training_rollout_time_limit_secs: usize,
    #[arg(long)]
    validation_rollout_time_limit_secs: usize,
    #[arg(long, default_value_t = 1)]
    num_python_tool_servers: usize,
    #[arg(long)]
    sglang_server_log_path: Option<String>,
    #[arg(long)]
    message_log_path: Option<String>,
    #[arg(long)]
    num_gpus: usize,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
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
        training_rollout_config_path,
        posterior_hyperparameters_path,
        num_total_epochs,
        ui,
        training_rollout_time_limit_secs,
        validation_rollout_time_limit_secs,
        num_python_tool_servers,
        sglang_server_log_path,
        message_log_path,
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
        training_config_common_path,
        training_time,
        num_iterations_limit,
        num_gpus,
    } = Args::parse();
    check_sympy_availability().unwrap();
    assert!(
        num_python_tool_servers > 0,
        "num_python_tool_servers must be positive"
    );

    if ui {
        ProgressTuiServer::initialize(message_log_path, |_command| {})
            .await
            .unwrap();
    }
    let validation_rollout_config: DirectRolloutConfig =
        read_json(validation_rollout_config_path).unwrap();
    let training_set_rollout_config: DirectRolloutConfig =
        read_json(training_rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let training_config_common: PythonTrainingConfigCommon =
        read_toml(training_config_common_path).unwrap();
    // do the rest of the orchestrator work here
    let client = reqwest::Client::new();
    let model_name = LlmModelName::from_str(&model_cli_name, true).unwrap();
    let progress_save_path =
        Orchestrator::progress_save_path(&model_name.cli_name(), &config_nickname);
    let progress = match read_json::<OrchestrationProgress>(&progress_save_path) {
        Ok(progress) => {
            log_info(format!(
                "Loaded progress from file: {}. Progress: {:?}",
                progress_save_path, progress
            ));
            progress
        }
        Err(_) => {
            log_info(format!(
                "No progress file found at: {}, starting fresh",
                progress_save_path
            ));
            OrchestrationProgress {
                status: OrchestrationStatus::WorkingOnValidation,
                epoch: 0,
                validation_accuracies: BTreeMap::new(),
            }
        }
    };
    let mut orchestrator = Orchestrator {
        config_nickname,
        validation_rollout_config,
        training_set_rollout_config,
        posterior_calculation_config,
        num_total_epochs,
        training_rollout_time_limit_secs,
        validation_rollout_time_limit_secs,
        num_python_tool_servers,
        client,
        max_rollout_concurrency,
        inference_server_handle: None,
        sglang_server_log_path,
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
        training_config_common,
        training_time,
        num_iterations_limit,
        progress,
        num_gpus,
    };

    let result = match model_name {
        LlmModelName::Gpt4o => orchestrator.orchestrate::<Gpt4o>().await,
        LlmModelName::Qwen3_06b => orchestrator.orchestrate::<Qwen3_06B>().await,
        LlmModelName::Qwen3_4b => orchestrator.orchestrate::<Qwen3_4B>().await,
        LlmModelName::Qwen25_7b => orchestrator.orchestrate::<Qwen25_7B>().await,
        LlmModelName::Qwen35_08b => orchestrator.orchestrate::<Qwen35_08B>().await,
        LlmModelName::Qwen35_4b => orchestrator.orchestrate::<Qwen35_4B>().await,
    };
    if let Err(e) = result {
        log_exit_hint(format!("Orchestrator exits with error: {}", e));
        log_warning(format!("Orchestrator exits with error: {}", e));
    }
    if ui {
        ProgressTuiServer::shutdown().await.unwrap();
    }
}
