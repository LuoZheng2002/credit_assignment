use std::backtrace::Backtrace;

use clap::{Parser, ValueEnum};
use proctitle::set_title;
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

use credit_assignment::{
    check_python_env::check_sympy_availability,
    directories::VALIDATION_ROLLOUT_CONFIG_PATH,
    directories::{
        inference_wrapper_log_path, text_logger_summary_path, text_logger_verbose_path,
        training_wrapper_log_path,
    },
    hybrid_dataset::Validation,
    json_toml_utils::read_json,
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelName, Mistral7BInstructV03, Qwen3_4B, Qwen3_06B,
        Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    orchestrator::{OrchestrationProgress, OrchestrationStatus, Orchestrator},
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    python_training_config::TrainingHyperparameters,
    rollout_config::{RolloutConfig, TrainingRolloutConfig},
    training_set::TrainingSetSortMode,
    utils::configure_mount_dir,
};
use research_utility::progress_text_logger::{
    ProgressTextLogger, log_error, log_exit_hint, log_info, log_window_name,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run direct tree rollout and save action logs"
)]
struct Args {
    #[arg(short = 'c', long)]
    config_path: String,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct OrchestratorConfig {
    model_cli_name: String,
    config_nickname: String,
    use_tool: bool,
    training_rollout_config_path: String,
    num_total_epochs: usize,
    training_hyperparameters: TrainingHyperparameters,
    num_iterations_limit: usize,
    training_rollout_secs: usize,
    validation_rollout_secs: usize,
    training_time: f32,
    num_gpus: usize,
    training_trajectory_len_cutoff: usize,
    total_time_limit_hours: f32,
    mount_dir: String,
    keep_action_logs: bool,
    positive_advantage_only: bool,
    training_set_sort_mode: TrainingSetSortMode,
}

fn ensure_parent_dir_exists(file_path: &str) -> Result<(), String> {
    let Some(parent) = Path::new(file_path).parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(parent).map_err(|err| {
        format!(
            "Failed to create parent directory {}: {}",
            parent.display(),
            err
        )
    })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
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
    let Args { config_path } = Args::parse();
    let config_contents = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read config file '{}': {}", config_path, err));
    let OrchestratorConfig {
        model_cli_name,
        config_nickname,
        use_tool,
        training_rollout_config_path,
        num_total_epochs,
        training_hyperparameters,
        training_rollout_secs,
        validation_rollout_secs,
        training_time,
        num_iterations_limit,
        num_gpus,
        training_trajectory_len_cutoff,
        mount_dir,
        keep_action_logs,
        positive_advantage_only,
        training_set_sort_mode,
        total_time_limit_hours: _, // mandatory in config, not yet wired into Orchestrator
    } = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));
    let process_title = format!("orchestrator_{}_{}", model_cli_name, config_nickname);
    set_title(&process_title);
    check_sympy_availability().unwrap();
    configure_mount_dir(&mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));
    let inference_wrapper_log_path =
        inference_wrapper_log_path(&mount_dir, &model_cli_name, &config_nickname);
    let training_wrapper_log_path =
        training_wrapper_log_path(&mount_dir, &model_cli_name, &config_nickname);
    let text_log_summary_path =
        text_logger_summary_path(&mount_dir, &model_cli_name, &config_nickname);
    let text_log_verbose_path =
        text_logger_verbose_path(&mount_dir, &model_cli_name, &config_nickname);
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare inference wrapper log directory: {}", err));
    ensure_parent_dir_exists(&training_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare training wrapper log directory: {}", err));
    ensure_parent_dir_exists(&text_log_summary_path)
        .unwrap_or_else(|err| panic!("failed to prepare text log summary directory: {}", err));
    ensure_parent_dir_exists(&text_log_verbose_path)
        .unwrap_or_else(|err| panic!("failed to prepare text log verbose directory: {}", err));
    assert!(num_gpus > 0, "--num-gpus must be positive");
    log_info(format!(
        "Local wrapper-managed inference/training will use num_gpus={}",
        num_gpus
    ));

    Orchestrator::write_config_paths_file(
        &model_cli_name,
        &config_nickname,
        &training_rollout_config_path,
        VALIDATION_ROLLOUT_CONFIG_PATH,
    )
    .unwrap_or_else(|err| panic!("failed to write config_paths.json: {}", err));

    ProgressTextLogger::initialize(text_log_summary_path.clone(), text_log_verbose_path.clone())
        .await
        .unwrap();
    log_window_name(format!(
        "Orchestrator Program. model: {}, config_nickname: {}",
        model_cli_name, config_nickname
    ));
    let validation_rollout_config: RolloutConfig<Validation> =
        read_json(VALIDATION_ROLLOUT_CONFIG_PATH).unwrap();
    let training_set_rollout_config: TrainingRolloutConfig =
        read_json(training_rollout_config_path).unwrap();
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )
    .unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    // do the rest of the orchestrator work here
    let client = reqwest::Client::new();
    let model_name = LlmModelName::from_str(&model_cli_name, true).unwrap();
    let progress_save_path =
        Orchestrator::progress_save_path(&mount_dir, &model_name.cli_name(), &config_nickname);
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
            let initial_status = OrchestrationStatus::WorkingOnValidation;
            OrchestrationProgress {
                status: initial_status,
                epoch: 0,
                validation_accuracies: BTreeMap::new(),
                training_rollout_accuracies: BTreeMap::new(),
                validation_rollout_llm_call_throughputs: BTreeMap::new(),
                training_rollout_llm_call_throughputs: BTreeMap::new(),
                training_throughputs: BTreeMap::new(),
            }
        }
    };
    let mut orchestrator = Orchestrator {
        config_nickname,
        validation_rollout_config,
        training_set_rollout_config,
        posterior_calculation_config,
        num_total_epochs,
        training_rollout_secs,
        validation_rollout_secs,
        client,
        inference_server_handle: None,
        inference_wrapper_log_path,
        training_wrapper_log_path,
        keep_action_logs,
        positive_advantage_only,
        training_hyperparameters,
        training_time,
        num_iterations_limit,
        progress,
        num_gpus,
        use_tool,
        mount_dir,
        training_set_sort_mode,
        training_trajectory_len_cutoff,
    };

    let result = match model_name {
        LlmModelName::Gemma3_4b => orchestrator.orchestrate::<Gemma3_4BIt>().await,
        LlmModelName::Llama31_8b => orchestrator.orchestrate::<Llama31_8BInstruct>().await,
        LlmModelName::Mistral7bInstructV03 => {
            orchestrator.orchestrate::<Mistral7BInstructV03>().await
        }
        LlmModelName::Qwen3_06b => orchestrator.orchestrate::<Qwen3_06B>().await,
        LlmModelName::Qwen3_4b => orchestrator.orchestrate::<Qwen3_4B>().await,
        LlmModelName::Qwen25_7b => orchestrator.orchestrate::<Qwen25_7B>().await,
        LlmModelName::Qwen35_08b => orchestrator.orchestrate::<Qwen35_08B>().await,
        LlmModelName::Qwen35_4b => orchestrator.orchestrate::<Qwen35_4B>().await,
    };
    if let Err(e) = result {
        log_exit_hint(format!("Orchestrator exits with error: {}", e));
        log_error(format!("Orchestrator exits with error: {}", e));
    }
    ProgressTextLogger::shutdown().await.unwrap();
}
