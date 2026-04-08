use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    call_llm::call_llm_chat_completions,
    deepmath::{
        generate_concise_reasoning::{DeepMathConciseReasoning, get_concise_reasoning_path},
        generate_raw_answers::{AnswerRaw, get_raw_answer_path},
        judge_answers::{DeepMathCorrectness, get_correctness_path},
    },
    parallel_process_jsonl::{HasId, parallel_process_jsonl},
};

#[derive(Serialize, Deserialize, Debug)]
pub struct ZippedDeepMathCorrectnessReasoning {
    pub id: usize,
    pub correct: bool,
    pub model_answer: String,
    pub correct_answer: String,
    pub question: String,
    pub model_reasoning: String,
    pub reference_reasoning: String,
}

impl HasId for ZippedDeepMathCorrectnessReasoning {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathErrorCause {
    pub id: usize,
    pub correct: bool,
    pub error_cause: Option<String>,
    pub question: Option<String>,
    pub model_reasoning: Option<String>,
}

impl HasId for DeepMathErrorCause {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_error_causes_path(
    model_name: &str,
    dataset_name: &str,
    num_samples: usize,
    is_rollout: bool,
) -> String {
    if is_rollout {
        format!(
            "results/{}/rollout/{}_error_causes_{}.jsonl",
            model_name, dataset_name, num_samples
        )
    } else {
        format!(
            "results/{}/{}_error_causes_{}.jsonl",
            model_name, dataset_name, num_samples
        )
    }
}

async fn generate_error_cause_task(
    correctness_reasoning: ZippedDeepMathCorrectnessReasoning,
    client: Client,
) -> DeepMathErrorCause {
    if correctness_reasoning.correct {
        return DeepMathErrorCause {
            id: correctness_reasoning.id,
            correct: true,
            error_cause: None,
            question: None,
            model_reasoning: None,
        };
    }
    let prompt = format!(
        "You are an error cause analyzer. You will be given the question, a model's reasoning process and the reference reasoning process. \
Your task is to find the core error cause in the model's reasoning that leads to the incorrect answer. \
If there are multiple errors, only state the first one. Use one sentence to summarize the error.\n\
Question: {}\nModel's Reasoning: {}\nReference Reasoning: {}\nError Cause Analysis:",
        correctness_reasoning.question,
        correctness_reasoning.model_reasoning,
        correctness_reasoning.reference_reasoning
    );
    let error_reason = call_llm_chat_completions(client, prompt, "gpt-5-mini").await;
    DeepMathErrorCause {
        id: correctness_reasoning.id,
        correct: correctness_reasoning.correct,
        error_cause: Some(error_reason),
        question: Some(correctness_reasoning.question),
        model_reasoning: Some(correctness_reasoning.model_reasoning),
    }
}

pub async fn generate_error_causes(
    model_name: &str,
    dataset_name: &str,
    num_samples: usize,
    client: Client,
    is_rollout: bool,
) {
    println!(
        "Generating error cause analysis for model {} on {} dataset with {} samples...",
        model_name, dataset_name, num_samples
    );
    let correctness_path = get_correctness_path(model_name, dataset_name, num_samples, is_rollout);
    let raw_answer_path = get_raw_answer_path(model_name, dataset_name, num_samples);
    let reference_reasoning_path = get_concise_reasoning_path(dataset_name, num_samples);
    let error_causes_output_path =
        get_error_causes_path(model_name, dataset_name, num_samples, is_rollout);
    parallel_process_jsonl(
        &[
            &correctness_path,
            &raw_answer_path,
            &reference_reasoning_path,
        ],
        &error_causes_output_path,
        |values| {
            assert_eq!(values.len(), 3);
            let correctness: DeepMathCorrectness = serde_json::from_value(values[0].clone())
                .expect(&format!("Failed to parse value: {}", values[0]));
            let raw_answer: AnswerRaw = serde_json::from_value(values[1].clone())
                .expect(&format!("Failed to parse value: {}", values[1]));
            let reference_reasoning: DeepMathConciseReasoning =
                serde_json::from_value(values[2].clone())
                    .expect(&format!("Failed to parse value: {}", values[2]));
            ZippedDeepMathCorrectnessReasoning {
                id: correctness.id,
                correct: correctness.correct,
                model_answer: correctness.model_answer,
                correct_answer: correctness.correct_answer,
                question: correctness.question,
                model_reasoning: raw_answer.model_reasoning,
                reference_reasoning: reference_reasoning.concise_reasoning,
            }
        },
        move |correctness_reasoning| {
            let client = client.clone();
            async move { generate_error_cause_task(correctness_reasoning, client).await }
        },
        2000,
    )
    .await
    .unwrap();
    println!(
        "Generated error cause analysis for model {} on {} DeepMath samples and saved to {}",
        model_name, num_samples, error_causes_output_path
    );
}
