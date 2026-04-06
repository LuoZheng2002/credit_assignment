use clap::Parser;
use credit_assignment::deepmath::generate_concise_reasoning::generate_concise_reasoning;
use reqwest::Client;
#[derive(Parser, Debug)]
#[command(name = "Generate Concise Reasoning")]
struct Args {
    #[arg(short, long)]
    num_samples: usize,
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    let Args { num_samples } = Args::parse();

    let client = Client::new();
    generate_concise_reasoning(num_samples, client).await;
}
