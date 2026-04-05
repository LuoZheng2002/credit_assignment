use reqwest::Client;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();

    let url = "http://localhost:8000/v1/chat/completions";

    let body = json!({
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "max_tokens": 100
    });

    let response = client.post(url).json(&body).send().await?;

    let json: serde_json::Value = response.json().await?;

    println!("{}", serde_json::to_string_pretty(&json)?);

    Ok(())
}
