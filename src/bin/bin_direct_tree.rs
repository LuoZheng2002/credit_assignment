use std::backtrace::Backtrace;

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::DirectRolloutConfig,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{
        Gpt4o, LlmCliArgs, LlmModelName, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
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
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    epoch: usize, // the epoch index
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long)]
    rollout_time_limit_secs: usize,
    #[arg(long, default_value_t = 1)]
    num_python_tool_servers: usize,
    #[arg(long)]
    sglang_server_log_path: Option<String>,
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
        llm_cli_args,
        max_rollout_concurrency,
        rollout_config_path,
        posterior_hyperparameters_path,
        epoch,
        ui,
        rollout_time_limit_secs,
        num_python_tool_servers,
        sglang_server_log_path,
    } = Args::parse();
    check_sympy_availability().unwrap();
    assert!(
        num_python_tool_servers > 0,
        "num_python_tool_servers must be positive"
    );

    println!("Starting direct rollout evaluation pipeline...");
    let client = Client::new();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&model_cli_name, true).unwrap();
    if ui {
        ProgressTuiServer::initialize(sglang_server_log_path.clone(), |_command| {})
            .await
            .unwrap();
    }
    let program_config = RolloutProgramConfig {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        client,
        max_rollout_concurrency,
        llm_cli_args,
        rollout_time_limit_secs,
        num_python_tool_servers,
    };
    match model_name {
        LlmModelName::Qwen25_7b => rollout_all::<Qwen25_7B>(program_config).await,
        LlmModelName::Qwen3_06b => rollout_all::<Qwen3_06B>(program_config).await,
        LlmModelName::Qwen3_4b => rollout_all::<Qwen3_4B>(program_config).await,
        LlmModelName::Qwen35_4b => rollout_all::<Qwen35_4B>(program_config).await,
        LlmModelName::Qwen35_08b => rollout_all::<Qwen35_08B>(program_config).await,
        LlmModelName::Gpt4o => rollout_all::<Gpt4o>(program_config).await,
    }
    if ui {
        ProgressTuiServer::shutdown().await.unwrap();
    }
}
