use std::backtrace::Backtrace;

use clap::{ArgAction, Parser, ValueEnum};
use proctitle::set_title;
use std::{collections::BTreeMap, path::Path};

use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        hybrid_dataset::{Training, Validation},
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
        rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
    },
    directories::{inference_wrapper_log_path, training_wrapper_log_path, tui_log_path},
    json_toml_utils::{read_json, read_toml},
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelName, Mistral7BInstructV03, Qwen3_4B, Qwen3_06B,
        Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    orchestrator::{OrchestrationProgress, OrchestrationStatus, Orchestrator},
    python_training_config::PythonTrainingConfigCommon,
    utils::configure_mount_dir,
    validation_config_path::VALIDATION_ROLLOUT_CONFIG_PATH,
};
use research_utility::progress_tui_logger::{
    ProgressTuiLogger, log_error, log_exit_hint, log_info, log_window_name,
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
    use_tool: bool,
    #[arg(long)]
    training_rollout_config_path: String,
    #[arg(long)]
    num_total_epochs: usize,
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
    max_python_processes: usize,
    #[arg(long)]
    num_gpus: usize,
    #[arg(long)]
    mount_dir: String,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    keep_action_logs: bool,
    #[arg(long, action = ArgAction::Set)]
    positive_advantage_only: bool,
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
    let Args {
        model_cli_name,
        config_nickname,
        max_rollout_concurrency,
        use_tool,
        training_rollout_config_path,
        num_total_epochs,
        ui,
        training_rollout_time_limit_secs,
        validation_rollout_time_limit_secs,
        max_python_processes,
        advantage_calculation_policy,
        training_config_common_path,
        training_time,
        num_iterations_limit,
        num_gpus,
        mount_dir,
        keep_action_logs,
        positive_advantage_only,
    } = Args::parse();
    let process_title = format!("orchestrator_{}_{}", model_cli_name, config_nickname);
    set_title(&process_title);
    check_sympy_availability().unwrap();
    assert!(
        max_python_processes > 0,
        "max_python_processes must be positive"
    );
    configure_mount_dir(&mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));
    let inference_wrapper_log_path = inference_wrapper_log_path(&model_cli_name, &config_nickname)
        .unwrap_or_else(|err| panic!("failed to render inference wrapper log path: {}", err));
    let training_wrapper_log_path = training_wrapper_log_path(&model_cli_name, &config_nickname)
        .unwrap_or_else(|err| panic!("failed to render training wrapper log path: {}", err));
    let tui_log_path = tui_log_path(&model_cli_name, &config_nickname)
        .unwrap_or_else(|err| panic!("failed to render tui log path: {}", err));
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare inference wrapper log directory: {}", err));
    ensure_parent_dir_exists(&training_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare training wrapper log directory: {}", err));
    ensure_parent_dir_exists(&tui_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare tui log directory: {}", err));
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

    if ui {
        ProgressTuiLogger::initialize(tui_log_path.clone())
            .await
            .unwrap();
        log_window_name(format!(
            "Orchestrator Program. model: {}, config_nickname: {}",
            model_cli_name, config_nickname
        ));
    }
    let validation_rollout_config: DirectRolloutConfig<Validation> =
        read_json(VALIDATION_ROLLOUT_CONFIG_PATH).unwrap();
    let training_set_rollout_config: DirectRolloutConfig<Training> =
        read_json(training_rollout_config_path).unwrap();
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::posterior_hyperparameters_path::posterior_hyperparameters_path(),
    )
    .unwrap();
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
        training_rollout_time_limit_secs,
        validation_rollout_time_limit_secs,
        max_python_processes,
        client,
        max_rollout_concurrency,
        inference_server_handle: None,
        inference_wrapper_log_path,
        training_wrapper_log_path,
        keep_action_logs,
        advantage_calculation_policy,
        positive_advantage_only,
        training_config_common,
        training_time,
        num_iterations_limit,
        progress,
        num_gpus,
        use_tool,
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
    if ui {
        ProgressTuiLogger::shutdown().await.unwrap();
    }
}
