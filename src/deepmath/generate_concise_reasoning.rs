use reqwest::Client;

use crate::{
    datasets::{DeepMathQuestionReasoning, get_deepmath_questions_with_reasoning_path},
    parallel_process_jsonl::{HasId, parallel_process_jsonl},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathConciseReasoning {
    pub id: usize,
    pub concise_reasoning: String,
    pub final_answer: String,
    pub question: String,
}

impl HasId for DeepMathConciseReasoning {
    fn id(&self) -> usize {
        self.id
    }
}

async fn generate_concise_reasoning_task(
    question: DeepMathQuestionReasoning,
    client: Client,
) -> DeepMathConciseReasoning {
    let prompt = format!(
        "You are a helpful assistant that provides concise reasoning steps for solving math problems. \
Given the following question, final answer and a reference reasoning paragraph, generate a concise reasoning process that leads to the final answer, with only the broad ideas and key intermediate results, without detailed calculations. \n\
Question: {}\nFinal Answer: {}\nReasoning: {}\nConcise Reasoning:",
        question.question, question.final_answer, question.reasoning
    );
    let body = serde_json::json!({
        "model": "gpt-5-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides concise reasoning steps for solving math problems."},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 4096,
    });
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .expect("Failed to send request to OpenAI API");
    let response_json: serde_json::Value = response
        .json()
        .await
        .expect("Failed to parse response from OpenAI API");
    let concise_reasoning = response_json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("content is invalid: {}", response_json))
        .trim()
        .to_string();
    DeepMathConciseReasoning {
        id: question.id,
        question: question.question,
        final_answer: question.final_answer,
        concise_reasoning,
    }
}

pub fn get_concise_reasoning_path(num_samples: usize) -> String {
    format!(
        "datasets/deepmath_samples_{}_concise_reasoning.jsonl",
        num_samples
    )
}

pub async fn generate_concise_reasoning(num_samples: usize, client: Client) {
    let questions_with_reasoning_path = get_deepmath_questions_with_reasoning_path(num_samples);
    let concise_reasoning_output_path = get_concise_reasoning_path(num_samples);
    parallel_process_jsonl(
        &[&questions_with_reasoning_path],
        &concise_reasoning_output_path,
        |values| {
            let question = serde_json::from_value::<DeepMathQuestionReasoning>(values[0].clone())
                .expect("Failed to parse question with reasoning");
            question
        },
        move |question| {
            let client = client.clone();
            async move { generate_concise_reasoning_task(question, client).await }
        },
        2000,
    )
    .await
    .unwrap();
    println!(
        "Generated concise reasoning for {} DeepMath samples and saved to {}",
        num_samples, concise_reasoning_output_path
    );
}
