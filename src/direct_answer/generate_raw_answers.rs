use crate::{
    call_llm::call_llm_chat_completions,
    datasets::{AssetFileDataset, DeepMathQuestion},
    llm_model::LlmModel,
    parallel_process_jsonl::{HasId, parallel_process_jsonl},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};



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
    let asset_file_dataset = AssetFileDataset {
        dataset: dataset_name.to_string(),
        num_samples,
    };
    let questions_path = asset_file_dataset.file_path();
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
