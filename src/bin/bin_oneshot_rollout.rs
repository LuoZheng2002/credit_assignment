use std::backtrace::Backtrace;
use std::path::Path;

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        hybrid_dataset::{DatasetSplit, DatasetSplitEnum, Testing, Training, Validation},
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
        rollout::{RolloutProgramConfig, rollout_all},
        rollout_config::DirectRolloutConfig,
    },
    jinja_directories::{
        action_logs_oneshot_path_from_template, inference_wrapper_log_path_from_template,
        model_parent_dir_from_template, rollout_summary_oneshot_path_from_template,
    },
    json_toml_utils::read_json,
    launch_inference_wrapper::{
        best_effort_shutdown_stale_inference_wrapper, launch_inference_wrapper_process,
        shut_down_inference_wrapper_process,
    },
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelMarker, LlmModelName, Mistral7BInstructV03,
        Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    utils::configure_mount_dir,
};
use reqwest::Client;
use research_utility::progress_tui_logger::ProgressTuiLogger;

const DEFAULT_PROGRESS_TUI_LOG_PATH: &str = "progress_tui_log.bin";

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run one-shot tree rollout and save action logs to one-shot paths"
)]
struct Args {
    #[arg(long)]
    model_cli_name: String,
    #[arg(long)]
    max_rollout_concurrency: usize,
    #[arg(long)]
    config_nickname_rollout: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long, value_enum)]
    dataset_split: DatasetSplitEnum,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long, default_value_t = 0)]
    epoch: usize,
    #[arg(long, default_value_t = 1)]
    total_epochs: usize,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long)]
    rollout_time_limit_secs: usize,
    #[arg(long, default_value_t = 1)]
    max_python_processes: usize,
    #[arg(long, default_value = DEFAULT_PROGRESS_TUI_LOG_PATH)]
    progress_tui_log_path: String,
    #[arg(long)]
    mount_dir: String,
    #[arg(long)]
    num_gpus: usize,
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

async fn run_rollout_for_split<M: LlmModelMarker, S: DatasetSplit>(
    rollout_config: DirectRolloutConfig<S>,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
    num_gpus: usize,
    inference_wrapper_log_path: &str,
) {
    best_effort_shutdown_stale_inference_wrapper().await;
    let model_parent_dir =
        model_parent_dir_from_template(M::CLI_NAME, &args.config_nickname_rollout, args.epoch)
            .unwrap_or_else(|err| panic!("failed to resolve model parent dir: {}", err));
    let model_path = format!("{}/model", model_parent_dir);

    let (sglang_port, mut process, listener_stop_signal, listener_handle) =
        launch_inference_wrapper_process(
            &model_path,
            M::CLI_NAME,
            &args.config_nickname_rollout,
            args.epoch,
            M::API_NAME,
            num_gpus,
            inference_wrapper_log_path,
        )
        .await
        .unwrap_or_else(|err| panic!("failed to launch inference server: {}", err));

    let action_log_store_override_path = action_logs_oneshot_path_from_template::<S>(
        &args.model_cli_name,
        &args.config_nickname_rollout,
        args.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to resolve one-shot action logs path: {}", err));

    let program_config = RolloutProgramConfig {
        config_nickname: args.config_nickname_rollout.clone(),
        rollout_config,
        posterior_calculation_config,
        epoch: args.epoch,
        client,
        max_rollout_concurrency: args.max_rollout_concurrency,
        inference_endpoint: credit_assignment::llm_model::InferenceEndpoint::SglangPort(
            sglang_port,
        ),
        rollout_time_limit_secs: args.rollout_time_limit_secs,
        max_python_processes: args.max_python_processes,
        total_epochs: args.total_epochs,
        action_log_store_override_path: Some(action_log_store_override_path),
    };
    let summary = rollout_all::<M, S>(program_config).await;

    let _ = listener_stop_signal.send(true);
    shut_down_inference_wrapper_process(&mut process).await;
    let _ = listener_handle.await;

    let summary_path = rollout_summary_oneshot_path_from_template(
        &args.model_cli_name,
        &args.config_nickname_rollout,
    )
    .unwrap_or_else(|err| panic!("failed to resolve rollout summary path: {}", err));
    if let Some(parent) = std::path::Path::new(&summary_path).parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("failed to create dir {}: {}", parent.display(), err));
    }
    let json = serde_json::to_string_pretty(&summary)
        .unwrap_or_else(|err| panic!("failed to serialize rollout summary: {}", err));
    std::fs::write(&summary_path, json).unwrap_or_else(|err| {
        panic!(
            "failed to write rollout summary to {}: {}",
            summary_path, err
        )
    });
    println!("Rollout summary written to {}", summary_path);
}

macro_rules! run_rollout {
    (
        $model_name:expr,
        $dataset_split:expr,
        $args:expr,
        $client:expr,
        $posterior:expr,
        $num_gpus:expr,
        $log_path:expr;
        $( $model_enum:path, $model_ty:ty ),+ $(,)?;
        $( $split_enum:path, $split_ty:ty ),+ $(,)?
    ) => {{
        let model_name = $model_name;
        let dataset_split = $dataset_split;
        let args = $args;
        let client = $client;
        let posterior = $posterior;
        let num_gpus = $num_gpus;
        let log_path = $log_path;

        macro_rules! run_model_for_split {
            ($rollout_config:expr, $inner_split_ty:ty) => {
                match model_name {
                    $(
                        $model_enum => {
                            run_rollout_for_split::<$model_ty, $inner_split_ty>(
                                $rollout_config,
                                args,
                                client,
                                posterior,
                                num_gpus,
                                log_path,
                            )
                            .await
                        }
                    ),+
                }
            };
        }

        match dataset_split {
            $(
                $split_enum => {
                    let rollout_config: DirectRolloutConfig<$split_ty> =
                        read_json(&args.rollout_config_path).unwrap();
                    run_model_for_split!(rollout_config, $split_ty)
                }
            ),+
        }
    }};
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
    assert!(args.total_epochs > 0, "total_epochs must be positive");
    assert!(args.num_gpus > 0, "--num-gpus must be positive");
    configure_mount_dir(&args.mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));

    println!("Starting one-shot rollout pipeline...");
    let client = Client::new();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(&args.posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&args.model_cli_name, true).unwrap();

    let inference_wrapper_log_path = inference_wrapper_log_path_from_template(
        &args.model_cli_name,
        &args.config_nickname_rollout,
    )
    .unwrap_or_else(|err| panic!("failed to render inference wrapper log path: {}", err));
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare inference wrapper log directory: {}", err));

    if args.ui {
        ProgressTuiLogger::initialize(args.progress_tui_log_path.clone())
            .await
            .unwrap();
    }

    run_rollout!(
        model_name,
        args.dataset_split,
        &args,
        client,
        posterior_calculation_config,
        args.num_gpus,
        &inference_wrapper_log_path;
        LlmModelName::Qwen25_7b, Qwen25_7B,
        LlmModelName::Qwen3_06b, Qwen3_06B,
        LlmModelName::Qwen3_4b, Qwen3_4B,
        LlmModelName::Qwen35_4b, Qwen35_4B,
        LlmModelName::Qwen35_08b, Qwen35_08B,
        LlmModelName::Gemma3_4b, Gemma3_4BIt,
        LlmModelName::Llama31_8b, Llama31_8BInstruct,
        LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03;
        DatasetSplitEnum::Training, Training,
        DatasetSplitEnum::Validation, Validation,
        DatasetSplitEnum::Testing, Testing
    );

    if args.ui {
        ProgressTuiLogger::shutdown().await.unwrap();
    }
}
