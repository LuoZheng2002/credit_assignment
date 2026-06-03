use std::backtrace::Backtrace;

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::DirectRolloutConfig,
        hybrid_dataset::{DatasetSplit, Testing, Training, Validation},
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
    dataset_split: DatasetSplitArg,
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

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DatasetSplitArg {
    Training,
    Validation,
    Testing,
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
    let Args {
        model_cli_name,
        rollout_config_path,
        dataset_split,
        posterior_hyperparameters_path,
        ui,
        sglang_server_log_path,
        ..
    } = &args;
    check_sympy_availability().unwrap();
    assert!(
        args.num_python_tool_servers > 0,
        "num_python_tool_servers must be positive"
    );
    assert!(args.total_epochs > 0, "total_epochs must be positive");

    println!("Starting direct rollout evaluation pipeline...");
    let client = Client::new();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(model_cli_name, true).unwrap();
    if *ui {
        ProgressTuiServer::initialize(sglang_server_log_path.clone(), |_command| {})
            .await
            .unwrap();
    }
    match dataset_split {
        DatasetSplitArg::Training => {
            let rollout_config: DirectRolloutConfig<Training> =
                read_json(rollout_config_path).unwrap();
            match model_name {
                LlmModelName::Qwen25_7b => {
                    run_rollout_for_split::<Qwen25_7B, Training>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen3_06b => {
                    run_rollout_for_split::<Qwen3_06B, Training>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen3_4b => {
                    run_rollout_for_split::<Qwen3_4B, Training>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen35_4b => {
                    run_rollout_for_split::<Qwen35_4B, Training>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen35_08b => {
                    run_rollout_for_split::<Qwen35_08B, Training>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Gpt4o => {
                    run_rollout_for_split::<Gpt4o, Training>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
            }
        }
        DatasetSplitArg::Validation => {
            let rollout_config: DirectRolloutConfig<Validation> =
                read_json(rollout_config_path).unwrap();
            match model_name {
                LlmModelName::Qwen25_7b => {
                    run_rollout_for_split::<Qwen25_7B, Validation>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen3_06b => {
                    run_rollout_for_split::<Qwen3_06B, Validation>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen3_4b => {
                    run_rollout_for_split::<Qwen3_4B, Validation>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen35_4b => {
                    run_rollout_for_split::<Qwen35_4B, Validation>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen35_08b => {
                    run_rollout_for_split::<Qwen35_08B, Validation>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Gpt4o => {
                    run_rollout_for_split::<Gpt4o, Validation>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
            }
        }
        DatasetSplitArg::Testing => {
            let rollout_config: DirectRolloutConfig<Testing> =
                read_json(rollout_config_path).unwrap();
            match model_name {
                LlmModelName::Qwen25_7b => {
                    run_rollout_for_split::<Qwen25_7B, Testing>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen3_06b => {
                    run_rollout_for_split::<Qwen3_06B, Testing>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen3_4b => {
                    run_rollout_for_split::<Qwen3_4B, Testing>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen35_4b => {
                    run_rollout_for_split::<Qwen35_4B, Testing>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Qwen35_08b => {
                    run_rollout_for_split::<Qwen35_08B, Testing>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
                LlmModelName::Gpt4o => {
                    run_rollout_for_split::<Gpt4o, Testing>(
                        rollout_config,
                        &args,
                        client,
                        posterior_calculation_config,
                    )
                    .await
                }
            }
        }
    }
    if *ui {
        ProgressTuiServer::shutdown().await.unwrap();
    }
}
