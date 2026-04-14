use clap::Parser;
use credit_assignment::apply_vllm_model_chat_template::apply_vllm_model_chat_template;
use credit_assignment::deepmath::generate_raw_answers::Model;
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
    let mut chat_template_prompt =
        apply_vllm_model_chat_template(Model::Qwen25_7b, &qwen_request.prompt, false);
    chat_template_prompt += &qwen_request.synthesized_response_prefix;
    let result = credit_assignment::call_llm::call_qwen_raw_completions(
        client,
        chat_template_prompt,
        Model::Qwen25_7b,
    )
    .await;
    println!("Qwen response:\n{}", result);
}
