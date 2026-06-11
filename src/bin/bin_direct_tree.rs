use std::backtrace::Backtrace;

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::DirectRolloutConfig,
        hybrid_dataset::{DatasetSplit, DatasetSplitEnum, Testing, Training, Validation},
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{
        Gemma3_4BIt, Gpt4o, Llama31_8BInstruct, LlmCliArgs, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
};
use reqwest::Client;
use research_utility::progress_tui_logger::ProgressTuiLogger;

const DEFAULT_PROGRESS_TUI_LOG_PATH: &str = "progress_tui_log.bin";

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run direct tree rollout and save action logs"
)]
struct Args {
    #[arg(long)]
    model_cli_name: String,
    #[command(flatten)]
    llm_cli_args: LlmCliArgs,
    #[arg(long)]
    max_rollout_concurrency: usize,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long, value_enum)]
    dataset_split: DatasetSplitEnum,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    epoch: usize, // the epoch index
    #[arg(long)]
    total_epochs: usize,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long)]
    rollout_time_limit_secs: usize,
    #[arg(long, default_value_t = 1)]
    max_python_processes: usize,
    #[arg(long)]
    inference_wrapper_log_path: Option<String>,
    #[arg(long, default_value = DEFAULT_PROGRESS_TUI_LOG_PATH)]
    progress_tui_log_path: String,
}

async fn run_rollout_for_split<M: LlmModelMarker, S: DatasetSplit>(
    rollout_config: DirectRolloutConfig<S>,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
) {
    let program_config = RolloutProgramConfig {
        config_nickname: args.config_nickname.clone(),
        rollout_config,
        posterior_calculation_config,
        epoch: args.epoch,
        client,
        max_rollout_concurrency: args.max_rollout_concurrency,
        llm_cli_args: args.llm_cli_args.clone(),
        rollout_time_limit_secs: args.rollout_time_limit_secs,
        max_python_processes: args.max_python_processes,
        total_epochs: args.total_epochs,
    };
    let _ = rollout_all::<M, S>(program_config).await;
}

macro_rules! run_rollout {
    (
        $model_name:expr,
        $dataset_split:expr,
        $args:expr,
        $client:expr,
        $posterior:expr;
        $( $model_enum:path, $model_ty:ty ),+ $(,)?;
        $( $split_enum:path, $split_ty:ty ),+ $(,)?
    ) => {{
        let model_name = $model_name;
        let dataset_split = $dataset_split;
        let args = $args;
        let client = $client;
        let posterior = $posterior;

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

    println!("Starting direct rollout evaluation pipeline...");
    let client = Client::new();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(&args.posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&args.model_cli_name, true).unwrap();
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
        posterior_calculation_config;
        LlmModelName::Qwen25_7b, Qwen25_7B,
        LlmModelName::Qwen3_06b, Qwen3_06B,
        LlmModelName::Qwen3_4b, Qwen3_4B,
        LlmModelName::Qwen35_4b, Qwen35_4B,
        LlmModelName::Qwen35_08b, Qwen35_08B,
        LlmModelName::Gemma3_4b, Gemma3_4BIt,
        LlmModelName::Llama31_8b, Llama31_8BInstruct,
        LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03,
        LlmModelName::Gpt4o, Gpt4o;
        DatasetSplitEnum::Training, Training,
        DatasetSplitEnum::Validation, Validation,
        DatasetSplitEnum::Testing, Testing
    );
    if args.ui {
        ProgressTuiLogger::shutdown().await.unwrap();
    }
}
