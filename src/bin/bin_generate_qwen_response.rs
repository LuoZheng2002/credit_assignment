use clap::Parser;
use credit_assignment::apply_vllm_model_chat_template::apply_vllm_model_chat_template;
use credit_assignment::call_llm::{LlmEndpoint, call_qwen_raw_completions_on_endpoint};
use credit_assignment::llm_model::LlmModel;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    #[arg(long)]
    pub vllm_port: u16,
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    let Args {
        input_file,
        vllm_port,
    } = Args::parse();
    assert!(vllm_port > 0, "--vllm-port must be greater than 0");
    let file_content = std::fs::read_to_string(input_file).expect("Failed to read input file");
    let qwen_request: QwenRequest =
        serde_json::from_str(&file_content).expect("Failed to parse JSON");
    let client = reqwest::Client::new();
    let mut chat_template_prompt =
        apply_vllm_model_chat_template(LlmModel::Qwen25_7b, &qwen_request.prompt, false);
    chat_template_prompt += &qwen_request.synthesized_response_prefix;
    let endpoint = Arc::new(LlmEndpoint::new(0, vllm_port, 1));
    let result = call_qwen_raw_completions_on_endpoint(
        client,
        chat_template_prompt,
        LlmModel::Qwen25_7b,
        endpoint,
    )
    .await;
    println!("Qwen response:\n{}", result);
}
