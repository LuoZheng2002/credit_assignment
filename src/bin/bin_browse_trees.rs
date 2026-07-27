use std::{backtrace::Backtrace, error::Error, io, path::PathBuf};

use clap::{Parser, ValueEnum};
use credit_assignment::browse_trees;
use credit_assignment::directories::{
    action_logs_oneshot_path, action_logs_path as standard_action_logs_path,
};
use credit_assignment::hybrid_dataset::{Testing, Training, Validation};
use credit_assignment::json_toml_utils::read_json;
use credit_assignment::posterior_calculation_config::PosteriorHyperparameters;
use crossterm::cursor::Show;
use crossterm::execute;
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ActionLogsSplit {
    Train,
    Validation,
    Testing,
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
    #[arg(long, default_value = "results")]
    mount_dir: String,
    #[arg(long)]
    oneshot: bool,
    #[arg(long)]
    override_hyperparameters_path: Option<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn resolved_action_logs_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    split: ActionLogsSplit,
) -> Result<PathBuf, String> {
    let path = match split {
        ActionLogsSplit::Train => standard_action_logs_path::<Training>(
            mount_dir,
            model_cli_name,
            config_nickname,
            epoch,
        )?,
        ActionLogsSplit::Validation => standard_action_logs_path::<Validation>(
            mount_dir,
            model_cli_name,
            config_nickname,
            epoch,
        )?,
        ActionLogsSplit::Testing => {
            standard_action_logs_path::<Testing>(mount_dir, model_cli_name, config_nickname, epoch)?
        }
    };
    Ok(repo_root().join(path))
}

fn oneshot_action_logs_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    split: ActionLogsSplit,
) -> Result<PathBuf, String> {
    let path = match split {
        ActionLogsSplit::Train => {
            action_logs_oneshot_path::<Training>(mount_dir, model_cli_name, config_nickname, epoch)?
        }
        ActionLogsSplit::Validation => action_logs_oneshot_path::<Validation>(
            mount_dir,
            model_cli_name,
            config_nickname,
            epoch,
        )?,
        ActionLogsSplit::Testing => {
            return Err("Testing split is not supported for one-shot action logs".to_string());
        }
    };
    Ok(repo_root().join(path))
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
        mount_dir,
        oneshot,
        override_hyperparameters_path,
    } = Args::parse();

    let action_logs_path = if oneshot {
        oneshot_action_logs_path(&mount_dir, &model_cli_name, &config_nickname, epoch, split)?
    } else {
        resolved_action_logs_path(&mount_dir, &model_cli_name, &config_nickname, epoch, split)?
    };

    if !action_logs_path.exists() {
        return Err(format!(
            "Missing action logs at {}. Auto-download is temporarily disabled; pass --mount-dir for the storage root or copy/symlink the rollout artifact into this exact path before browsing.",
            action_logs_path.display()
        )
        .into());
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
