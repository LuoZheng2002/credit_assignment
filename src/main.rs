use candle_vllm::api::{EngineBuilder, ModelRepo};
use candle_vllm::openai::requests::{ChatCompletionRequest, ChatMessage, Messages};

#[tokio::main]
async fn main() -> Result<(), String> {
    println!("Building engine...");
    let engine = EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen2.5-7B", None)))
        .build_async()
        .await?;
    println!("Engine built successfully!");

    let request = ChatCompletionRequest {
        model: Some("default".to_string()),
        messages: Messages::Map(vec![std::collections::HashMap::from([
            ("role".to_string(), "user".to_string()),
            (
                "content".to_string(),
                "Say hello from the Rust API.".to_string(),
            ),
        ])]),
        max_tokens: Some(64),
        ..Default::default()
    };
    println!("Sending request...");
    let response = engine.generate_request(request).await.map_err(|e| e.to_string())?;
    println!("Response received: {:?}", response);
    engine.shutdown();
    Ok(())
}
