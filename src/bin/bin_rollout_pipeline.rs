use clap::Parser;
use credit_assignment::{
    call_llm::set_vllm_port,
    deepmath::{generate_raw_answers::Model, judge_answers::judge_answers},
    multi_agent::{
        generate_rollout_answers::generate_rollout_answers,
        parse_rollout_answers::parse_rollout_answers,
    },
};
use pyo3::Python;
use rand::{SeedableRng, rngs::StdRng};
use reqwest::Client;

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: Model,
    #[arg(long)]
    vllm_port: u16,
}

#[tokio::main]
async fn main() {
    println!("Starting rollout evaluation pipeline...");
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    assert!(
        std::env::var("PYTHONPATH").is_ok(),
        "PYTHONPATH environment variable is not set"
    );
    Python::initialize();
    let Args {
        dataset_name,
        model,
        num_samples,
        vllm_port,
    } = Args::parse();
    assert!(vllm_port > 0, "--vllm-port must be greater than 0");
    set_vllm_port(vllm_port);
    let model_name = model.cli_name();
    println!(
        "Evaluating model {} on {} dataset with {} samples (vLLM port: {})",
        model_name, dataset_name, num_samples, vllm_port
    );

    let client = Client::new();
    let mut rng = StdRng::seed_from_u64(42);
    let verifier_probability = 1.0;
    generate_rollout_answers(
        &dataset_name,
        num_samples,
        client.clone(),
        model,
        verifier_probability,
        &mut rng,
    )
    .await;

    parse_rollout_answers(model, &dataset_name, num_samples).await;

    judge_answers(model, &dataset_name, num_samples, client.clone(), true).await;

    // generate_error_causes(model_name, &dataset_name, num_samples, client, true).await;
}
