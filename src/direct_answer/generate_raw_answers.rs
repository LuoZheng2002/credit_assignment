use crate::{
    call_llm::call_llm_chat_completions,
    datasets::{DeepMathQuestion, get_question_path},
    parallel_process_jsonl::{HasId, parallel_process_jsonl},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(clap::ValueEnum, Clone, Debug, Copy)]
pub enum LlmModel {
    #[value(name = "gpt-4o")]
    Gpt4o,
    #[value(name = "gpt-5-mini")]
    Gpt5Mini,
    #[value(name = "qwen2.5-7b")]
    Qwen25_7b,
    #[value(name = "qwen3-4b")]
    Qwen3_4b,
    #[value(name = "qwen3-8b")]
    Qwen3_8b,
    #[value(name = "qwen3.5-4b")]
    Qwen35_4b,
}

impl LlmModel {
    pub fn cli_name(&self) -> &'static str {
        match self {
            LlmModel::Gpt4o => "gpt-4o",
            LlmModel::Gpt5Mini => "gpt-5-mini",
            LlmModel::Qwen25_7b => "qwen2.5-7b",
            LlmModel::Qwen3_4b => "qwen3-4b",
            LlmModel::Qwen3_8b => "qwen3-8b",
            LlmModel::Qwen35_4b => "qwen3.5-4b",
        }
    }

    pub fn api_name(&self) -> &'static str {
        match self {
            LlmModel::Gpt4o => "gpt-4o",
            LlmModel::Gpt5Mini => "gpt-5-mini",
            LlmModel::Qwen25_7b => "Qwen/Qwen2.5-7B-Instruct",
            LlmModel::Qwen3_4b => "Qwen/Qwen3-4B",
            LlmModel::Qwen3_8b => "Qwen/Qwen3-8B",
            LlmModel::Qwen35_4b => "Qwen/Qwen3.5-4B",
        }
    }

    pub fn is_qwen(&self) -> bool {
        match self {
            LlmModel::Qwen25_7b | LlmModel::Qwen3_4b | LlmModel::Qwen3_8b | LlmModel::Qwen35_4b => true,
            LlmModel::Gpt4o | LlmModel::Gpt5Mini => false,
        }
    }

    pub fn is_gpt(&self) -> bool {
        match self {
            LlmModel::Gpt4o | LlmModel::Gpt5Mini => true,
            LlmModel::Qwen25_7b | LlmModel::Qwen3_4b | LlmModel::Qwen3_8b | LlmModel::Qwen35_4b => false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AnswerRaw {
    pub id: usize,
    pub correct_answer: String,
    pub model_reasoning: String,
    pub question: String,
}

impl HasId for AnswerRaw {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_raw_answer_path(model: LlmModel, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/{}_raw_{}.jsonl",
        model.cli_name(),
        dataset_name,
        num_samples
    )
}

async fn generate_raw_answer_task(
    question: DeepMathQuestion,
    client: Client,
    model: LlmModel,
) -> AnswerRaw {
    let prompt = format!(
        "Please answer the following question by first reasoning and then putting the final short answer in \\boxed{{}}. Question: {}",
        question.question
    );
    let response = call_llm_chat_completions(client, prompt, model, false).await;
    AnswerRaw {
        id: question.id,
        question: question.question,
        model_reasoning: response,
        correct_answer: question.final_answer,
    }
}

pub async fn generate_raw_answers(
    dataset_name: &str,
    num_samples: usize,
    client: Client,
    model: LlmModel,
) {
    let questions_path = get_question_path(dataset_name, num_samples);
    let raw_output_path = get_raw_answer_path(model, dataset_name, num_samples);
    parallel_process_jsonl(
        &[&questions_path],
        &raw_output_path,
        |values| {
            let question = serde_json::from_value::<DeepMathQuestion>(values[0].clone())
                .expect("Failed to parse question");
            question
        },
        move |question| {
            let client = client.clone();
            async move { generate_raw_answer_task(question, client, model).await }
        },
        2000,
    )
    .await
    .unwrap();
}
