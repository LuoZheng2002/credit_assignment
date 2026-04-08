use clap::Parser;
use credit_assignment::apply_qwen_chat_template::apply_qwen_chat_template;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct QwenRequest {
    pub prompt: String,
    pub synthesized_response_prefix: String,
}

#[derive(Parser, Debug)]
#[command(name = "Generate Qwen Response")]
pub struct Args {
    #[arg(short, long)]
    pub input_file: String,
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    let Args { input_file } = Args::parse();
    let file_content = std::fs::read_to_string(input_file).expect("Failed to read input file");
    let qwen_request: QwenRequest =
        serde_json::from_str(&file_content).expect("Failed to parse JSON");
    let client = reqwest::Client::new();
    let mut chat_template_prompt = apply_qwen_chat_template(&qwen_request.prompt);
    chat_template_prompt += &qwen_request.synthesized_response_prefix;
    let model_name = "Qwen/Qwen2.5-7B-Instruct";
    let result = credit_assignment::call_llm::call_qwen_raw_completions(
        client,
        chat_template_prompt,
        model_name,
    )
    .await;
    println!("Qwen response:\n{}", result);
}
