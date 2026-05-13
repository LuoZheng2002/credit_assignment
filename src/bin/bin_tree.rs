use std::backtrace::Backtrace;

use clap::Parser;
use credit_assignment::{
    agent::rollout_batch::rollout_batch,
    direct_answer::generate_raw_answers::LlmModel,
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
    #[arg(long)]
    vllm_port: u16,
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
        vllm_port,
    } = Args::parse();
    rollout_batch(model, dataset_name, num_samples, vllm_port).await;
}
