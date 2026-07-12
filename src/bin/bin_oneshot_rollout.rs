use std::backtrace::Backtrace;
use std::path::Path;

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    hybrid_dataset::{DatasetSplit, DatasetSplitEnum, Testing, Training, Validation},
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
    training_set::generate_training_trajectories_with_path,
    tree_action_log::ActionLogStore,
    directories::{
        action_logs_oneshot_path, inference_wrapper_log_path, model_parent_dir,
        rollout_summary_oneshot_path, training_trajectories_oneshot_path,
        training_trajectories_stats_oneshot_path,
    },
    fixed_temperatures,
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
use ordered_float::NotNan;
use reqwest::Client;
use research_utility::progress_tui_logger::{ProgressTuiLogger, log_info};

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
    #[arg(long)]
    use_tool: bool,
    #[arg(long, value_enum)]
    dataset_split: DatasetSplitEnum,
    #[arg(long, default_value_t = 0)]
    epoch: usize,
    #[arg(long, default_value_t = 1)]
    total_epochs: usize,
    #[arg(long, value_enum)]
    advantage_calculation_policy: AdvantageCalculationPolicy,
    #[arg(long, action = ArgAction::Set)]
    positive_advantage_only: bool,
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
    let action_log_store_override_path = action_logs_oneshot_path::<S>(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
        args.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to resolve one-shot action logs path: {}", err));

    // Check if previous run already exhausted the time limit; if so, skip the
    // inference server startup and rollout entirely.
    let prev_elapsed =
        ActionLogStore::<M, S>::initialize_if_missing(&action_log_store_override_path)
            .and_then(|store| store.read_elapsed_time())
            .unwrap_or(0.0);
    if prev_elapsed >= args.rollout_time_limit_secs as f32 {
        log_info(format!(
            "Skipping rollout: previous elapsed time ({prev_elapsed:.1}s) already meets \
             or exceeds the time limit ({}s). No new actions will be added.",
            args.rollout_time_limit_secs
        ));
        return;
    }

    best_effort_shutdown_stale_inference_wrapper().await;
    let model_parent_dir = model_parent_dir(
        &args.mount_dir,
        M::CLI_NAME,
        &args.config_nickname_rollout,
        args.epoch,
    );
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
        use_tool: args.use_tool,
        fixed_temperature: NotNan::new(if S::IS_TRAINING {
            fixed_temperatures::TRAINING_TEMPERATURE
        } else {
            fixed_temperatures::VALIDATION_TEMPERATURE
        })
        .unwrap(),
    };
    let summary = rollout_all::<M, S>(&args.mount_dir, program_config).await;

    let _ = listener_stop_signal.send(true);
    shut_down_inference_wrapper_process(&mut process).await;
    let _ = listener_handle.await;

    let summary_path = rollout_summary_oneshot_path(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
    );
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

macro_rules! generate_trajectories {
    (
        $model_name:expr,
        $action_log_store_path:expr,
        $trajectories_dir:expr,
        $trajectories_msgpack_path:expr,
        $stats_path:expr,
        $config_bundle_path:expr,
        $rollout_config:expr,
        $posterior_calculation_config:expr,
        $advantage_calculation_policy:expr,
        $positive_advantage_only:expr,
        $use_tool:expr;
        $( $model_enum:path, $model_ty:ty ),+ $(,)?
    ) => {
        match $model_name {
            $(
                $model_enum => {
                    generate_training_trajectories_with_path::<$model_ty>(
                        $action_log_store_path,
                        $trajectories_dir,
                        $trajectories_msgpack_path,
                        $stats_path,
                        $config_bundle_path,
                        $rollout_config,
                        $posterior_calculation_config,
                        $advantage_calculation_policy,
                        $positive_advantage_only,
                        $use_tool,
                    )
                    .await
                }
            ),+
        }
    };
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
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )
    .unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&args.model_cli_name, true).unwrap();

    let inference_wrapper_log_path = inference_wrapper_log_path(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
    );
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare inference wrapper log directory: {}", err));

    if args.ui {
        ProgressTuiLogger::initialize(args.progress_tui_log_path.clone())
            .await
            .unwrap();
    }

    let posterior_calculation_config_clone = posterior_calculation_config.clone();

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

    // Generate training trajectories from one-shot action logs (only for Training split)
    if matches!(args.dataset_split, DatasetSplitEnum::Training) {
        println!("Generating training trajectories from one-shot action logs...");
        let trajectories_dir = training_trajectories_oneshot_path(
            &args.mount_dir,
            &args.model_cli_name,
            &args.config_nickname_rollout,
        );
        let trajectories_msgpack_path = format!("{}/trajectories.msgpack", trajectories_dir);
        let stats_path = training_trajectories_stats_oneshot_path(
            &args.mount_dir,
            &args.model_cli_name,
            &args.config_nickname_rollout,
        );
        let config_bundle_path = format!("{}/config_bundle.json", trajectories_dir);
        let action_log_store_path = action_logs_oneshot_path::<Training>(
            &args.mount_dir,
            &args.model_cli_name,
            &args.config_nickname_rollout,
            args.epoch,
        )
        .unwrap_or_else(|err| panic!("failed to resolve action logs path: {}", err));
        let rollout_config: DirectRolloutConfig<Training> =
            read_json(&args.rollout_config_path).unwrap();

        generate_trajectories!(
            model_name,
            &action_log_store_path,
            &trajectories_dir,
            &trajectories_msgpack_path,
            &stats_path,
            &config_bundle_path,
            rollout_config,
            posterior_calculation_config_clone,
            args.advantage_calculation_policy,
            args.positive_advantage_only,
            args.use_tool;
            LlmModelName::Qwen25_7b, Qwen25_7B,
            LlmModelName::Qwen3_06b, Qwen3_06B,
            LlmModelName::Qwen3_4b, Qwen3_4B,
            LlmModelName::Qwen35_4b, Qwen35_4B,
            LlmModelName::Qwen35_08b, Qwen35_08B,
            LlmModelName::Gemma3_4b, Gemma3_4BIt,
            LlmModelName::Llama31_8b, Llama31_8BInstruct,
            LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03
        );
        println!(
            "Training trajectories generated at {}",
            trajectories_msgpack_path
        );
    }

    if args.ui {
        ProgressTuiLogger::shutdown().await.unwrap();
    }
}
