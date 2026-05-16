use clap::Parser;
use credit_assignment::direct_answer::{
    generate_error_causes::generate_error_causes,
    generate_raw_answers::generate_raw_answers,
    judge_answers::judge_answers,
    parse_answers::parse_answers,
};
use credit_assignment::llm_model::LlmModel;
use reqwest::Client;

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: LlmModel,
}

#[tokio::main]
async fn main() {
    println!("Starting DeepMath evaluation pipeline...");
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
    let model_name = model.cli_name();
    println!(
        "Evaluating model {} on {} dataset with {} samples",
        model_name, dataset_name, num_samples
    );

    let client = Client::new();
    generate_raw_answers(&dataset_name, num_samples, client.clone(), model).await;

    parse_answers(model, &dataset_name, num_samples).await;

    judge_answers(model, &dataset_name, num_samples, client.clone(), false).await;

    generate_error_causes(model, &dataset_name, num_samples, client, false).await;
}
