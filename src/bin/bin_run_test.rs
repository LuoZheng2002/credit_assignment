use std::backtrace::Backtrace;

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::DirectRolloutConfig,
        direct_tree_action_log::AssetFileDirectTreeActionLogs,
        hybrid_dataset::Testing,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    get_accuracy::{TestAccuracyResult, get_test_accuracies},
    json_line_util::{read_json, write_json},
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
    about = "Run test rollout and compute per-dataset accuracies with confidence intervals"
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
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    epoch: usize,
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

fn model_cli_name_to_string(model_name: &LlmModelName) -> String {
    match model_name {
        LlmModelName::Gpt4o => Gpt4o::CLI_NAME,
        LlmModelName::Qwen3_06b => Qwen3_06B::CLI_NAME,
        LlmModelName::Qwen3_4b => Qwen3_4B::CLI_NAME,
        LlmModelName::Qwen25_7b => Qwen25_7B::CLI_NAME,
        LlmModelName::Qwen35_08b => Qwen35_08B::CLI_NAME,
        LlmModelName::Qwen35_4b => Qwen35_4B::CLI_NAME,
    }
    .to_string()
}

async fn run_rollout_and_compute_accuracy<M: LlmModelMarker>(
    rollout_config: DirectRolloutConfig<Testing>,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
) -> TestAccuracyResult {
    let program_config = RolloutProgramConfig {
        config_nickname: args.config_nickname.clone(),
        rollout_config: rollout_config.clone(),
        posterior_calculation_config: posterior_calculation_config.clone(),
        epoch: args.epoch,
        client,
        max_rollout_concurrency: args.max_rollout_concurrency,
        llm_cli_args: args.llm_cli_args.clone(),
        rollout_time_limit_secs: args.rollout_time_limit_secs,
        num_python_tool_servers: args.num_python_tool_servers,
        total_epochs: args.total_epochs,
    };
    rollout_all::<M, Testing>(program_config).await;

    let asset_file_action_logs = AssetFileDirectTreeActionLogs::<M, Testing> {
        nickname: args.config_nickname.clone(),
        rollout_config: rollout_config.clone(),
        posterior_calculation_config: posterior_calculation_config,
        epoch: args.epoch,
        _phantom: std::marker::PhantomData,
    };
    get_test_accuracies::<M, Testing>(
        asset_file_action_logs,
        "Test accuracy",
        rollout_config.max_num_trunks,
    )
    .await
}

macro_rules! run_model_for_testing {
    ($model_name:expr, $rollout_config:expr, $args:expr, $client:expr, $posterior:expr;
     $( $model_enum:path, $model_ty:ty ),+ $(,)?) => {{
        let model_name = $model_name;
        let rollout_config = $rollout_config;
        let args = $args;
        let client = $client;
        let posterior = $posterior;

        match model_name {
            $(
                $model_enum => {
                    run_rollout_and_compute_accuracy::<$model_ty>(
                        rollout_config,
                        args,
                        client,
                        posterior,
                    )
                    .await
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
        args.num_python_tool_servers > 0,
        "num_python_tool_servers must be positive"
    );
    assert!(args.total_epochs > 0, "total_epochs must be positive");

    println!("Starting test accuracy evaluation pipeline...");
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
    let model_cli_name = model_cli_name_to_string(&model_name);
    let rollout_config: DirectRolloutConfig<Testing> =
        read_json::<DirectRolloutConfig<Testing>>(&args.rollout_config_path).unwrap();
    assert_eq!(
        rollout_config.max_num_trunks, rollout_config.max_num_total_trajectories,
        "max_num_trunks ({}) must equal max_num_total_trajectories ({}) for test evaluation (no branching)",
        rollout_config.max_num_trunks, rollout_config.max_num_total_trajectories,
    );

    let test_result = run_model_for_testing!(
        model_name,
        rollout_config,
        &args,
        client,
        posterior_calculation_config;
        LlmModelName::Qwen25_7b, Qwen25_7B,
        LlmModelName::Qwen3_06b, Qwen3_06B,
        LlmModelName::Qwen3_4b, Qwen3_4B,
        LlmModelName::Qwen35_4b, Qwen35_4B,
        LlmModelName::Qwen35_08b, Qwen35_08B,
        LlmModelName::Gpt4o, Gpt4o
    );

    let output_path = format!(
        "results/{}/{}/test_accuracy_epoch_{}.json",
        model_cli_name, args.config_nickname, args.epoch
    );
    write_json(&output_path, &test_result).unwrap();
    println!("Test accuracy results written to {}", output_path);

    if args.ui {
        ProgressTuiServer::shutdown().await.unwrap();
    }
}