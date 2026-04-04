use candle_vllm::{api::{Engine, EngineBuilder, ModelRepo}, openai::requests::{ChatCompletionRequest, Messages}};

pub enum EvalEngine {
    Vllm(Engine),
    Reqwest {
        client: reqwest::Client,
        url: String,
        model_name: String,
    },
}


impl EvalEngine {
    pub async fn new_vllm(model_repo: ModelRepo) -> Self {
        let engine = EngineBuilder::new(model_repo)
            .build_async()
            .await
            .expect("Failed to build VLLM engine");
        EvalEngine::Vllm(engine)
    }
    pub async fn call(&self, input: &str) -> Result<String, String> {
        match self {
            EvalEngine::Vllm(engine) => {
                let response = engine.generate_request(ChatCompletionRequest{
                    messages: Messages::Map(vec![std::collections::HashMap::from([
                        ("role".to_string(), "user".to_string()),
                        ("content".to_string(), input.to_string()),
                    ])]),
                    max_tokens: Some(64),
                    ..Default::default()
                }).await.map_err(|e| e.to_string())?;
                let first_choice = response.choices.first().ok_or("No choices in response")?;
                let content = first_choice.message.content.clone().ok_or("No content in first choice")?;
                Ok(content)
            }
            EvalEngine::Reqwest { client, url, model_name } => {
                let request_body = serde_json::json!({
                    "model": model_name,
                    "input": input,
                });
                // let response = client.post(url)
                //     .json(&request_body)
                //     .send()
                //     .await?
                //     .json::<serde_json::Value>()
                //     .await?;
                // Ok(response["output"].as_str().unwrap_or_default().to_string())
                unimplemented!("Reqwest-based engine is not implemented yet")
            }
        }
    }
}