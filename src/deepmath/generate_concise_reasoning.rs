use reqwest::Client;

use crate::{
    call_llm::call_llm, datasets::{DeepMathQuestionReasoning, get_questions_with_reasoning_path}, parallel_process_jsonl::{HasId, parallel_process_jsonl}
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
    let concise_reasoning = call_llm(client, prompt, "gpt-5-mini").await;
    DeepMathConciseReasoning {
        id: question.id,
        question: question.question,
        final_answer: question.final_answer,
        concise_reasoning,
    }
}

pub fn get_concise_reasoning_path(dataset_name: &str, num_samples: usize) -> String {
    format!(
        "datasets/{}_samples_{}_concise_reasoning.jsonl",
        dataset_name, num_samples
    )
}

pub async fn generate_concise_reasoning(dataset_name: &str, num_samples: usize, client: Client) {
    let questions_with_reasoning_path = get_questions_with_reasoning_path(dataset_name, num_samples);
    let concise_reasoning_output_path = get_concise_reasoning_path(dataset_name, num_samples);
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
