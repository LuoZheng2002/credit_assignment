use crate::{
    call_llm::call_llm_chat_completions,
    datasets::{DeepMathQuestion, get_question_path},
    parallel_process_jsonl::{HasId, parallel_process_jsonl},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(clap::ValueEnum, Clone, Debug, Copy)]
pub enum Model {
    #[value(name = "gpt")]
    Gpt,
    #[value(name = "qwen")]
    Qwen,
}

impl Model {
    pub fn name(&self) -> &'static str {
        match self {
            Model::Gpt => "gpt",
            Model::Qwen => "qwen",
        }
    }
    pub fn full_name(&self) -> &'static str {
        match self {
            Model::Gpt => "gpt-4o",
            Model::Qwen => "Qwen/Qwen2.5-7B-Instruct",
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

pub fn get_raw_answer_path(model_name: &str, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/{}_raw_{}.jsonl",
        model_name, dataset_name, num_samples
    )
}

async fn generate_raw_answer_task(
    question: DeepMathQuestion,
    client: Client,
    model: Model,
) -> AnswerRaw {
    let prompt = format!(
        "Please answer the following question by first reasoning and then putting the final short answer in \\boxed{{}}. Question: {}",
        question.question
    );
    let model_name = match model {
        Model::Gpt => "gpt-4o",
        Model::Qwen => "Qwen/Qwen2.5-7B-Instruct",
    };
    let response = call_llm_chat_completions(client, prompt, model_name).await;
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
    model: Model,
) {
    let questions_path = get_question_path(dataset_name, num_samples);
    let raw_output_path = get_raw_answer_path(model.name(), dataset_name, num_samples);
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
