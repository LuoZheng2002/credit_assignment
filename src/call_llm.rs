use reqwest::Client;

use crate::apply_vllm_model_chat_template::apply_vllm_model_chat_template;
use crate::deepmath::generate_raw_answers::Model;

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
const VLLM_CHAT_COMPLETIONS_URL: &str = "http://localhost:8000/v1/chat/completions";
const VLLM_RAW_COMPLETIONS_URL: &str = "http://localhost:8000/v1/completions";

fn get_chat_completions_url(model: Model) -> &'static str {
    if model.is_gpt() {
        OPENAI_CHAT_COMPLETIONS_URL
    } else if model.is_qwen() {
        VLLM_CHAT_COMPLETIONS_URL
    } else {
        panic!("Unsupported model family for {}", model.api_name());
    }
}

fn build_chat_completions_body(prompt: String, model: Model) -> serde_json::Value {
    if model.is_qwen() {
        // Explicitly disable Qwen3 thinking mode in vLLM chat templating.
        return serde_json::json!({
            "model": model.api_name(),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_completion_tokens": 2048,
            "stop": ["<tool_wait>"],
            "chat_template_kwargs": {
                "enable_thinking": false,
            }
        });
    }

    serde_json::json!({
        "model": model.api_name(),
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_completion_tokens": 2048,
        "stop": ["<tool_wait>"],
    })
}

async fn post_json(
    client: Client,
    url: &str,
    body: serde_json::Value,
    model: Model,
) -> reqwest::Response {
    if model.is_gpt() {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
        return client
            .post(url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .unwrap();
    }

    client.post(url).json(&body).send().await.unwrap()
}

pub async fn call_llm_chat_completions(client: Client, prompt: String, model: Model) -> String {
    let url = get_chat_completions_url(model);
    let body = build_chat_completions_body(prompt, model);
    let response = post_json(client, url, body, model).await;
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
    assert!(
        model.is_qwen(),
        "call_qwen_raw_completions only supports Qwen-family models",
    );
    let body = serde_json::json!({
        "model": model.api_name(),
        "prompt": chat_template_prompt,
        "max_tokens": 2048,
        "stop": ["<tool_wait>"],
        "include_stop_str_in_output": true,
    });

    let response = post_json(client, VLLM_RAW_COMPLETIONS_URL, body, model).await;
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
        let mut planner_chat_template_prompt =
            apply_vllm_model_chat_template(model, &prompt_before_assistant, false);
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
