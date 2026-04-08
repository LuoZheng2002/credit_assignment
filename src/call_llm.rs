use reqwest::Client;

pub async fn call_llm(client: Client, prompt: String, model_name: &str) -> String {
    let url = if model_name.to_lowercase().contains("gpt") {
        "https://api.openai.com/v1/chat/completions"
    } else if model_name.to_lowercase().contains("qwen") {
        "http://localhost:8000/v1/chat/completions"
    } else {
        panic!("Unsupported model name: {}", model_name);
    };
    let body = serde_json::json!({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_completion_tokens": 2048,
    });
    let response = if model_name.to_lowercase().contains("gpt") {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
        client
            .post(url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .unwrap()
    } else {
        client.post(url).json(&body).send().await.unwrap()
    };
    let json: serde_json::Value = response.json().await.unwrap();
    json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("LLM response is invalid: {:?}", json))
        .to_string()
}
