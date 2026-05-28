use std::{backtrace::Backtrace, sync::Arc};

use clap::{ArgAction, Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::DirectRolloutConfig,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{Gpt4o, LlmCliArgs, LlmModelName, Qwen3_4B, Qwen25, Qwen35_4B, Qwen35_08B},
};
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};
use crossterm::{cursor::Show, event::DisableMouseCapture, execute};
use pyo3::Python;
use reqwest::Client;
use research_utility::progress_screen::ProgressScreen;
use tokio::sync::Semaphore;

fn restore_terminal_after_panic() {
    let _ = disable_raw_mode();
    let mut stderr = std::io::stderr();
    let _ = execute!(stderr, LeaveAlternateScreen, DisableMouseCapture, Show);
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run direct tree rollout and save action logs"
)]
struct Args {
    #[command(flatten)]
    llm_cli_args: LlmCliArgs,
    #[arg(long)]
    max_concurrent_questions: usize,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    epoch: usize,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long)]
    first_n_samples: Option<usize>,
    #[arg(long, default_value_t = 1)]
    max_sqlite_connections: u32,
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        restore_terminal_after_panic();
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
        config_nickname,
        llm_cli_args,
        max_concurrent_questions,
        rollout_config_path,
        posterior_hyperparameters_path,
        epoch,
        ui,
        first_n_samples,
        max_sqlite_connections,
    } = Args::parse();
    Python::initialize();
    check_sympy_availability().unwrap();

    println!("Starting direct rollout evaluation pipeline...");
    let client = Client::new();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    if rollout_config.accuracy_under_temperature.is_none() {
        eprintln!(
            "WARNING: rollout_config.accuracy_under_temperature is None; all segment posteriors will use mean=0 and std=1."
        );
    }
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let question_semaphore = Arc::new(Semaphore::new(max_concurrent_questions));
    let model_name = LlmModelName::from_str(&llm_cli_args.model_cli_name, true).unwrap();
    if ui {
        ProgressScreen::initialize(
            "Bin Direct Tree Rollout Progress",
            true,
            Some("log/log.txt"),
        )
        .await
        .unwrap();
    }
    let program_config = RolloutProgramConfig {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        client,
        question_semaphore,
        llm_cli_args,
        first_n_samples,
        max_sqlite_connections,
    };
    match model_name {
        LlmModelName::Qwen25_7b => rollout_all::<Qwen25>(program_config).await,
        LlmModelName::Qwen3_4b => rollout_all::<Qwen3_4B>(program_config).await,
        LlmModelName::Qwen35_4b => rollout_all::<Qwen35_4B>(program_config).await,
        LlmModelName::Qwen35_08b => rollout_all::<Qwen35_08B>(program_config).await,
        LlmModelName::Gpt4o => rollout_all::<Gpt4o>(program_config).await,
    }
    if ui {
        ProgressScreen::shutdown().await.unwrap();
    }
}
