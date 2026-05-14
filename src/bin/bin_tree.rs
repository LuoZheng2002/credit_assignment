use std::backtrace::Backtrace;

use clap::Parser;
use credit_assignment::{
    agent::rollout_batch::rollout_batch,
    direct_answer::generate_raw_answers::LlmModel,
    progress_screen::ProgressScreenConfig,
    progress_screen::ProgressScreen,
    worker_message_tx::{clear_worker_message_tx, set_worker_message_tx},
};

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: LlmModel,
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    vllm_ports: Vec<u16>,
}

// we want to log each action
// if a question finishes, record the trajectory immediately
// For loading, first load trajectories and find finished question indices, then remove all logs related to these questions
// Then reconstruct unfinished trajectories from logs and continue the rollout
// If all trajectories finish, sort trajectories and report final overall tree correctness accuracy.

#[tokio::main]
async fn main() {
    println!("Starting rollout evaluation pipeline...");
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
        dataset_name,
        model,
        num_samples,
        vllm_ports,
    } = Args::parse();

    let mut progress_screen_config = ProgressScreenConfig::from_defaults(vllm_ports.len(), 1);
    progress_screen_config.window_title = "Bin Tree Rollout Progress".to_string();
    progress_screen_config.key_order = vec![
        "status".to_string(),
        "model".to_string(),
        "dataset".to_string(),
        "num_samples".to_string(),
        "endpoints".to_string(),
        "running_accuracy".to_string(),
    ];
    progress_screen_config.persist_after_channel_close = false;

    let progress_screen = ProgressScreen::new(progress_screen_config);
    set_worker_message_tx(progress_screen.clone_message_tx());

    rollout_batch(model, dataset_name, num_samples, vllm_ports).await;

    clear_worker_message_tx();
    drop(progress_screen);
}
