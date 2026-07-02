use std::{backtrace::Backtrace, path::Path};

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        hybrid_dataset::Testing,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
        rollout::{RolloutProgramConfig, rollout_all},
        rollout_config::DirectRolloutConfig,
        tree_action_log::open_action_logs,
    },
    get_accuracy::{TestAccuracyResult, get_test_accuracies},
    jinja_directories::{
        inference_wrapper_log_path_from_template, model_parent_dir_from_template,
        test_accuracy_path_from_template, tui_log_path_from_template,
    },
    json_toml_utils::{read_json, write_json},
    launch_inference_wrapper::{
        best_effort_shutdown_stale_inference_wrapper, launch_inference_wrapper_process,
        shut_down_inference_wrapper_process,
    },
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    utils::configure_mount_dir,
};
use reqwest::Client;
use research_utility::progress_tui_logger::ProgressTuiLogger;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run test rollout and compute per-dataset accuracies with confidence intervals"
)]
struct Args {
    #[arg(long)]
    testing_configs_path: String,

    #[arg(long)]
    max_rollout_concurrency: usize,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long)]
    rollout_time_limit_secs: usize,
    #[arg(long, default_value_t = 1)]
    max_python_processes: usize,
    #[arg(long, default_value_t = 1)]
    num_gpus: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestingConfig {
    model_cli_name: String,
    config_nickname: String,
    testing_rollout_config_path: String,
    posterior_hyperparameters_path: String,
    epoch: usize,
    total_epochs: usize,
    mount_dir: String,
}

fn model_cli_name_to_string(model_name: &LlmModelName) -> String {
    match model_name {
        LlmModelName::Gemma3_4b => Gemma3_4BIt::CLI_NAME,
        LlmModelName::Llama31_8b => Llama31_8BInstruct::CLI_NAME,
        LlmModelName::Mistral7bInstructV03 => Mistral7BInstructV03::CLI_NAME,
        LlmModelName::Qwen3_06b => Qwen3_06B::CLI_NAME,
        LlmModelName::Qwen3_4b => Qwen3_4B::CLI_NAME,
        LlmModelName::Qwen25_7b => Qwen25_7B::CLI_NAME,
        LlmModelName::Qwen35_08b => Qwen35_08B::CLI_NAME,
        LlmModelName::Qwen35_4b => Qwen35_4B::CLI_NAME,
    }
    .to_string()
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

async fn run_rollout_and_compute_accuracy<M: LlmModelMarker>(
    rollout_config: DirectRolloutConfig<Testing>,
    testing_config: &TestingConfig,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
    inference_endpoint: InferenceEndpoint,
) -> TestAccuracyResult {
    let program_config = RolloutProgramConfig {
        config_nickname: testing_config.config_nickname.clone(),
        rollout_config: rollout_config.clone(),
        posterior_calculation_config: posterior_calculation_config.clone(),
        epoch: testing_config.epoch,
        client,
        max_rollout_concurrency: args.max_rollout_concurrency,
        inference_endpoint,
        rollout_time_limit_secs: args.rollout_time_limit_secs,
        max_python_processes: args.max_python_processes,
        total_epochs: testing_config.total_epochs,
    };
    let _ = rollout_all::<M, Testing>(program_config).await;

    let _ = open_action_logs::<M, Testing>(&testing_config.config_nickname, testing_config.epoch);
    get_test_accuracies::<M, Testing>(
        testing_config.config_nickname.clone(),
        rollout_config.clone(),
        posterior_calculation_config,
        testing_config.epoch,
        "Test accuracy",
        rollout_config.max_num_trunks,
    )
    .await
}

async fn run_rollout_and_compute_accuracy_with_server<M: LlmModelMarker>(
    rollout_config: DirectRolloutConfig<Testing>,
    testing_config: &TestingConfig,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
    inference_wrapper_log_path: &str,
) -> Result<TestAccuracyResult, String> {
    best_effort_shutdown_stale_inference_wrapper().await;
    let model_parent_dir = model_parent_dir_from_template(
        M::CLI_NAME,
        &testing_config.config_nickname,
        testing_config.epoch,
    )?;
    let model_path = format!("{}/model", model_parent_dir);
    let (sglang_port, mut process, listener_stop_signal, listener_handle) =
        launch_inference_wrapper_process(
            &model_path,
            M::CLI_NAME,
            &testing_config.config_nickname,
            testing_config.epoch,
            M::API_NAME,
            args.num_gpus,
            inference_wrapper_log_path,
        )
        .await?;

    let test_result = run_rollout_and_compute_accuracy::<M>(
        rollout_config,
        testing_config,
        args,
        client,
        posterior_calculation_config,
        InferenceEndpoint::SglangPort(sglang_port),
    )
    .await;

    let _ = listener_stop_signal.send(true);
    shut_down_inference_wrapper_process(&mut process).await;
    let _ = listener_handle.await;
    Ok(test_result)
}

macro_rules! run_model_for_testing {
    ($model_name:expr, $rollout_config:expr, $testing_config:expr, $args:expr, $client:expr, $posterior:expr,
     $inference_wrapper_log_path:expr;
     $( $model_enum:path, $model_ty:ty ),+ $(,)?) => {{
        let model_name = $model_name;
        let rollout_config = $rollout_config;
        let testing_config = $testing_config;
        let args = $args;
        let client = $client;
        let posterior = $posterior;

        match model_name {
            $(
                $model_enum => {
                    run_rollout_and_compute_accuracy_with_server::<$model_ty>(
                        rollout_config,
                        testing_config,
                        args,
                        client,
                        posterior,
                        $inference_wrapper_log_path,
                    )
                    .await
                }
            ),+
        }
    }};
}

async fn run_testing_config(
    testing_config: &TestingConfig,
    args: &Args,
    client: Client,
) -> Result<(), String> {
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(&testing_config.posterior_hyperparameters_path)?;
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&testing_config.model_cli_name, true)
        .map_err(|err| err.to_string())?;
    let model_cli_name = model_cli_name_to_string(&model_name);
    configure_mount_dir(&testing_config.mount_dir)?;
    let inference_wrapper_log_path =
        inference_wrapper_log_path_from_template(&model_cli_name, &testing_config.config_nickname)
            .map_err(|err| format!("failed to render inference wrapper log path: {}", err))?;
    let progress_tui_log_path =
        tui_log_path_from_template(&model_cli_name, &testing_config.config_nickname)
            .map_err(|err| format!("failed to render tui log path: {}", err))?;
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .map_err(|err| format!("failed to prepare inference wrapper log directory: {}", err))?;
    ensure_parent_dir_exists(&progress_tui_log_path)
        .map_err(|err| format!("failed to prepare tui log directory: {}", err))?;

    let rollout_config: DirectRolloutConfig<Testing> =
        read_json::<DirectRolloutConfig<Testing>>(&testing_config.testing_rollout_config_path)?;
    assert_eq!(
        rollout_config.max_num_trunks, rollout_config.max_num_total_trajectories,
        "max_num_trunks ({}) must equal max_num_total_trajectories ({}) for test evaluation (no branching)",
        rollout_config.max_num_trunks, rollout_config.max_num_total_trajectories,
    );

    if args.ui {
        ProgressTuiLogger::initialize(progress_tui_log_path.clone())
            .await
            .map_err(|err| err.to_string())?;
    }

    let result = async {
        let test_result = run_model_for_testing!(
            model_name,
            rollout_config,
            testing_config,
            args,
            client,
            posterior_calculation_config,
            &inference_wrapper_log_path;
            LlmModelName::Qwen25_7b, Qwen25_7B,
            LlmModelName::Qwen3_06b, Qwen3_06B,
            LlmModelName::Qwen3_4b, Qwen3_4B,
            LlmModelName::Qwen35_4b, Qwen35_4B,
            LlmModelName::Qwen35_08b, Qwen35_08B,
            LlmModelName::Gemma3_4b, Gemma3_4BIt,
            LlmModelName::Llama31_8b, Llama31_8BInstruct,
            LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03
        )?;

        let output_path = test_accuracy_path_from_template(
            &model_cli_name,
            &testing_config.config_nickname,
            testing_config.epoch,
        )
        .map_err(|err| {
            format!(
                "failed to render test accuracy path for model_cli_name={}, config_nickname={}, epoch={}: {}",
                model_cli_name,
                testing_config.config_nickname,
                testing_config.epoch,
                err,
            )
        })?;
        write_json(&output_path, &test_result)?;
        println!("Test accuracy results written to {}", output_path);
        Ok::<(), String>(())
    }
    .await;

    if args.ui {
        if let Err(err) = ProgressTuiLogger::shutdown()
            .await
            .map_err(|err| err.to_string())
        {
            if result.is_ok() {
                return Err(err);
            }
            eprintln!(
                "warning: progress TUI shutdown failed after config failure: {}",
                err
            );
        }
    }

    result
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
    let args = Args::parse();
    check_sympy_availability().unwrap();
    assert!(
        args.max_python_processes > 0,
        "max_python_processes must be positive"
    );
    assert!(args.num_gpus > 0, "num_gpus must be positive");

    println!("Starting test accuracy evaluation pipeline...");
    let client = Client::new();
    let testing_configs = read_json::<Vec<TestingConfig>>(&args.testing_configs_path)
        .unwrap_or_else(|err| panic!("failed to read testing configs: {}", err));
    assert!(
        !testing_configs.is_empty(),
        "testing_configs_path must contain at least one testing config"
    );

    for (index, testing_config) in testing_configs.iter().enumerate() {
        println!(
            "Starting testing config {} of {}: model_cli_name={}, config_nickname={}, epoch={}, total_epochs={}, mount_dir={}",
            index + 1,
            testing_configs.len(),
            testing_config.model_cli_name,
            testing_config.config_nickname,
            testing_config.epoch,
            testing_config.total_epochs,
            testing_config.mount_dir,
        );
        if let Err(err) = run_testing_config(testing_config, &args, client.clone()).await {
            panic!(
                "testing config {} of {} failed for model_cli_name={}, config_nickname={}, epoch={}, total_epochs={}: {}",
                index + 1,
                testing_configs.len(),
                testing_config.model_cli_name,
                testing_config.config_nickname,
                testing_config.epoch,
                testing_config.total_epochs,
                err,
            );
        }
    }
}
