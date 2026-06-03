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
        Gpt4o, LlmCliArgs, LlmModelMarker, LlmModelName, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B,
        Qwen35_08B,
    },
};
use reqwest::Client;
use research_utility::progress_tui_server::ProgressTuiServer;

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
    num_python_tool_servers: usize,
    #[arg(long)]
    sglang_server_log_path: Option<String>,
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
        num_python_tool_servers: args.num_python_tool_servers,
        total_epochs: args.total_epochs,
    };
    rollout_all::<M, S>(program_config).await;
}

macro_rules! run_model_for_split {
    ($model_name:expr, $rollout_config:expr, $args:expr, $client:expr, $posterior:expr, $split:ty) => {
        match $model_name {
            LlmModelName::Qwen25_7b => {
                run_rollout_for_split::<Qwen25_7B, $split>(
                    $rollout_config,
                    $args,
                    $client,
                    $posterior,
                )
                .await
            }
            LlmModelName::Qwen3_06b => {
                run_rollout_for_split::<Qwen3_06B, $split>(
                    $rollout_config,
                    $args,
                    $client,
                    $posterior,
                )
                .await
            }
            LlmModelName::Qwen3_4b => {
                run_rollout_for_split::<Qwen3_4B, $split>(
                    $rollout_config,
                    $args,
                    $client,
                    $posterior,
                )
                .await
            }
            LlmModelName::Qwen35_4b => {
                run_rollout_for_split::<Qwen35_4B, $split>(
                    $rollout_config,
                    $args,
                    $client,
                    $posterior,
                )
                .await
            }
            LlmModelName::Qwen35_08b => {
                run_rollout_for_split::<Qwen35_08B, $split>(
                    $rollout_config,
                    $args,
                    $client,
                    $posterior,
                )
                .await
            }
            LlmModelName::Gpt4o => {
                run_rollout_for_split::<Gpt4o, $split>(
                    $rollout_config,
                    $args,
                    $client,
                    $posterior,
                )
                .await
            }
        }
    };
}

async fn run_for_dataset_split<S: DatasetSplit>(
    model_name: LlmModelName,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
) {
    let rollout_config: DirectRolloutConfig<S> = read_json(&args.rollout_config_path).unwrap();
    run_model_for_split!(
        model_name,
        rollout_config,
        args,
        client,
        posterior_calculation_config,
        S
    );
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
        args.num_python_tool_servers > 0,
        "num_python_tool_servers must be positive"
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
        ProgressTuiServer::initialize(args.sglang_server_log_path.clone(), |_command| {})
            .await
            .unwrap();
    }
    match args.dataset_split {
        DatasetSplitEnum::Training => {
            run_for_dataset_split::<Training>(model_name, &args, client, posterior_calculation_config)
                .await
        }
        DatasetSplitEnum::Validation => {
            run_for_dataset_split::<Validation>(
                model_name,
                &args,
                client,
                posterior_calculation_config,
            )
            .await
        }
        DatasetSplitEnum::Testing => {
            run_for_dataset_split::<Testing>(model_name, &args, client, posterior_calculation_config)
                .await
        }
    }
    if args.ui {
        ProgressTuiServer::shutdown().await.unwrap();
    }
}
