use crate::{
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
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathAnswerRaw {
    pub id: usize,
    pub correct_answer: String,
    pub model_reasoning: String,
    pub question: String,
}

impl HasId for DeepMathAnswerRaw {
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
) -> DeepMathAnswerRaw {
    let prompt = format!(
        "Please answer the following question by first reasoning and then putting the final short answer in \\boxed{{}}. Question: {}",
        question.question
    );
    let (url, model_name) = match model {
        Model::Gpt => ("https://api.openai.com/v1/chat/completions", "gpt-4o"),
        Model::Qwen => (
            "http://localhost:8000/v1/chat/completions",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
    };
    let body = serde_json::json!({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_completion_tokens": 4096,
    });

    let response = match model {
        Model::Qwen => client.post(url).json(&body).send().await.unwrap(),
        Model::Gpt => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .expect("OPENAI_API_KEY environment variable not set");
            client
                .post(url)
                .bearer_auth(api_key)
                .json(&body)
                .send()
                .await
                .unwrap()
        }
    };
    let json: serde_json::Value = response.json().await.unwrap();
    let model_reasoning = json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("model answer is invalid: {:?}", json))
        .to_string();
    DeepMathAnswerRaw {
        id: question.id,
        question: question.question,
        model_reasoning,
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
