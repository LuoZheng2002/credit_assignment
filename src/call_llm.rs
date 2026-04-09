use reqwest::Client;

pub async fn call_llm_chat_completions(client: Client, prompt: String, model_name: &str) -> String {
    let url = if model_name.contains("gpt") {
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
        "stop": ["<tool_wait>"],
        // "include_stop_str_in_output": true,
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
    let body = response.bytes().await.unwrap();
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) else {
        panic!(
            "Failed to parse LLM response as JSON. Response text: {:?}",
            String::from_utf8_lossy(&body)
        );
    };
    json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("LLM response is invalid: {:?}", json))
        .to_string()
}

pub async fn call_qwen_raw_completions(
    client: Client,
    chat_template_prompt: String,
    model_name: &str,
) -> String {
    assert!(
        model_name.to_lowercase().contains("qwen"),
        "call_qwen_raw_completions only supports Qwen-family models",
    );
    let url = "http://localhost:8000/v1/completions";
    let body = serde_json::json!({
        "model": model_name,
        "prompt": chat_template_prompt,
        "max_tokens": 2048,
        "stop": ["<tool_wait>"],
        "include_stop_str_in_output": true,
    });

    let response = client.post(url).json(&body).send().await.unwrap();
    let json: serde_json::Value = response.json().await.unwrap();
    json["choices"][0]["text"]
        .as_str()
        .expect(&format!("Qwen completions response is invalid: {:?}", json))
        .to_string()
}
