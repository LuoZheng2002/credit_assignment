use reqwest::Client;

use crate::apply_qwen_chat_template::apply_qwen_chat_template;
use crate::deepmath::generate_raw_answers::Model;

pub async fn call_llm_chat_completions(client: Client, prompt: String, model: Model) -> String {
    let model_name = model.api_name();
    let url = if model.is_gpt() {
        "https://api.openai.com/v1/chat/completions"
    } else if model.is_qwen() {
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
    let response = if model.is_gpt() {
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
    model: Model,
) -> String {
    let model_name = model.api_name();
    assert!(
        model.is_qwen(),
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

pub async fn call_llm_with_prefix(
    client: Client,
    prompt_before_assistant: String,
    prompt_after_assistant: String,
    model: Model,
) -> String {
    if model.is_qwen() {
        let mut planner_chat_template_prompt = apply_qwen_chat_template(&prompt_before_assistant);
        planner_chat_template_prompt += &prompt_after_assistant;
        call_qwen_raw_completions(client.clone(), planner_chat_template_prompt, model).await
    } else {
        let full_prompt = format!(
            "{}\nAssistant: {}",
            prompt_before_assistant, prompt_after_assistant
        );
        call_llm_chat_completions(client.clone(), full_prompt, model).await
    }
}
