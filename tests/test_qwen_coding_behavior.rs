use std::fs;

use credit_assignment::{apply_qwen_chat_template::{ChatMessage, apply_qwen_chat_template}, call_llm::{call_llm_chat_completions, call_qwen_raw_completions}};
use minijinja::{Environment, context};
use serde::Serialize;
use tokenizers::Tokenizer;

#[tokio::test]
async fn test_qwen_template() {
    let prompt = "What is the capital of France?";

    let rendered = apply_qwen_chat_template(prompt);
    println!("{}", rendered);
}

#[tokio::test]
async fn test_qwen_coding_behavior() {
    let prompt = r#"Please find the sum of all prime numbers within 10000.
You can invoke python code by putting it in a markdown code block starting with ```python and ending with ```.
Put the final result in \boxed{}."#;
    let rendered = apply_qwen_chat_template(prompt);
    println!("{}", rendered);
    let client = reqwest::Client::new();
    let model_name = "Qwen/Qwen2.5-7B-Instruct";
    let result = call_qwen_raw_completions(client, rendered, model_name).await;
    println!("Qwen response: {}", result);
}
