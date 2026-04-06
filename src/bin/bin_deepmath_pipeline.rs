use clap::Parser;
use credit_assignment::deepmath::{generate_error_causes::generate_error_causes, generate_raw_answers::{Model, generate_raw_answers}, judge_answers::judge_answers, parse_answers::parse_answers};
use reqwest::Client;

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: Model,
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
    let Args { model, num_samples } = Args::parse();
    let model_name = model.name();
    println!(
        "Evaluating model {} on DeepMath dataset with {} samples",
        model_name, num_samples
    );
    
    let client = Client::new();
    generate_raw_answers(num_samples, client.clone(), model).await;

    parse_answers(model_name, num_samples).await;

    judge_answers(model, num_samples, client.clone()).await;

    generate_error_causes(model_name, num_samples, client).await;
}
