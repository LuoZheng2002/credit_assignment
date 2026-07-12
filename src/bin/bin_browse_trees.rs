use std::{
    backtrace::Backtrace,
    error::Error,
    io,
    path::{Path, PathBuf},
    process::Stdio,
};

use clap::{Parser, ValueEnum};
use credit_assignment::browse_trees;
use credit_assignment::posterior_calculation_config::PosteriorHyperparameters;
use credit_assignment::json_toml_utils::read_json;
use crossterm::cursor::Show;
use crossterm::execute;
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};
use tokio::process::Command;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ActionLogsSplit {
    Train,
    Validation,
}

impl ActionLogsSplit {
    fn script_arg(self) -> &'static str {
        match self {
            ActionLogsSplit::Train => "train",
            ActionLogsSplit::Validation => "validation",
        }
    }

    fn artifact_name(self) -> &'static str {
        match self {
            ActionLogsSplit::Train => "action_logs_training.extsort",
            ActionLogsSplit::Validation => "action_logs_validation.extsort",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Interactively browse rollout session logs, downloading action logs if needed"
)]
struct Args {
    #[arg(long)]
    model_cli_name: String,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    epoch: usize,
    #[arg(long, value_enum)]
    split: ActionLogsSplit,
    #[arg(long)]
    override_hyperparameters_path: Option<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn action_logs_path(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    split: ActionLogsSplit,
) -> PathBuf {
    repo_root()
        .join("results")
        .join("medium_files")
        .join(model_cli_name)
        .join(config_nickname)
        .join(format!("epoch_{}", epoch))
        .join(split.artifact_name())
}

async fn download_action_logs(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    split: ActionLogsSplit,
    destination: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("scripts/download_action_logs.py")
        .arg("--model-cli-name")
        .arg(model_cli_name)
        .arg("--config-nickname")
        .arg(config_nickname)
        .arg("--epoch")
        .arg(epoch.to_string())
        .arg("--split")
        .arg(split.script_arg())
        .current_dir(repo_root())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command.status().await?;

    if !status.success() {
        return Err(format!(
            "download_action_logs.py failed with status {} while fetching {}",
            status,
            destination.display()
        )
        .into());
    }

    if !destination.exists() {
        return Err(format!(
            "download_action_logs.py completed but the expected action log path is still missing: {}",
            destination.display()
        )
        .into());
    }

    Ok(())
}

fn restore_terminal_after_panic() {
    let _ = disable_raw_mode();
    let mut stderr = io::stderr();
    let _ = execute!(
        stderr,
        LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture,
        Show
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
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
        model_cli_name,
        config_nickname,
        epoch,
        split,
        override_hyperparameters_path,
    } = Args::parse();

    let action_logs_path = action_logs_path(&model_cli_name, &config_nickname, epoch, split);

    if !action_logs_path.exists() {
        eprintln!(
            "Missing action logs for {}/{}/epoch_{} ({:?}); downloading...",
            model_cli_name, config_nickname, epoch, split
        );
        download_action_logs(
            &model_cli_name,
            &config_nickname,
            epoch,
            split,
            &action_logs_path,
        )
        .await?;
    }

    let override_hyperparameters = override_hyperparameters_path
        .map(|path| read_json::<PosteriorHyperparameters>(path).unwrap());

    let result = browse_trees::run(action_logs_path, override_hyperparameters).await;

    let _ = disable_raw_mode();
    let mut stderr = io::stderr();
    let _ = execute!(
        stderr,
        LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture,
        Show
    );
    result
}
