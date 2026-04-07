use clap::Parser;
use credit_assignment::{
    deepmath::{
        generate_error_causes::generate_error_causes, generate_raw_answers::Model,
        judge_answers::judge_answers,
    },
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
}

#[tokio::main]
async fn main() {
    println!("Starting rollout evaluation pipeline...");
    Python::initialize();
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    // load env from .env file
    dotenvy::dotenv().ok();
    let Args {
        dataset_name,
        model,
        num_samples,
    } = Args::parse();
    let model_name = model.name();
    println!(
        "Evaluating model {} on {} dataset with {} samples",
        model_name, dataset_name, num_samples
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

    parse_rollout_answers(model_name, &dataset_name, num_samples).await; // modify this to be parse_rollout_answers

    judge_answers(model_name, &dataset_name, num_samples, client.clone(), true).await;

    generate_error_causes(model_name, &dataset_name, num_samples, client, true).await;
}
