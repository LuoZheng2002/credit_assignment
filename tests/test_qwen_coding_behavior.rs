use std::fs;

use credit_assignment::{
    apply_qwen_chat_template::{ChatMessage, apply_qwen_chat_template},
    call_llm::{call_llm_chat_completions, call_qwen_raw_completions},
};
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
    let mut rendered = apply_qwen_chat_template(prompt);
    rendered += r#"To find the sum of all prime numbers within 10000, we can write a Python script to generate all prime numbers up to 10000 and then sum them up. Let's start by writing the code to achieve this.
```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

sum_of_primes = sum(i for i in range(10001) if is_prime(i))
sum_of_primes
```
"#;
    println!("{}", rendered);
    let client = reqwest::Client::new();
    let model_name = "Qwen/Qwen2.5-7B-Instruct";
    let result = call_qwen_raw_completions(client, rendered, model_name).await;
    println!("Qwen response: {}", result);
}
