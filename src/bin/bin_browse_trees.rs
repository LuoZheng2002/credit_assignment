use std::backtrace::Backtrace;
use std::error::Error;
use std::io;

use clap::Parser;
use credit_assignment::browse_trees;
use credit_assignment::direct_tool::posterior_calculation_config::PosteriorHyperparameters;
use credit_assignment::json_toml_utils::read_json;
use crossterm::cursor::Show;
use crossterm::event::EnableMouseCapture;
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactively browse rollout session logs")]
struct Args {
    #[arg(long)]
    action_logs_path: String,
    #[arg(long)]
    override_hyperparameters_path: Option<String>,
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
        action_logs_path,
        override_hyperparameters_path,
    } = Args::parse();

    let override_hyperparameters = override_hyperparameters_path
        .map(|path| read_json::<PosteriorHyperparameters>(path).unwrap());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = browse_trees::run(action_logs_path, &mut terminal, override_hyperparameters).await;

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
