use std::path::Path;

use clap::Parser;
use credit_assignment::parallel_process_jsonl::{HasId, parallel_process_jsonl};
use reqwest::Client;
use serde::{Deserialize, Serialize};

const INPUT_FOLDER: &str = "datasets/deepmath_samples";
const OUTPUT_FOLDER_PREFIX: &str = "results/deepmath_samples";

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    num_samples: usize,
    #[arg(value_enum, short, long)]
    model: Model,
}

#[derive(clap::ValueEnum, Clone, Debug, Copy)]
enum Model {
    #[value(name = "gpt")]
    Gpt5Mini,
    #[value(name = "qwen")]
    Qwen25Instruct,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeepMathQuestion {
    id: usize,
    question: String,
    final_answer: String,
}

impl HasId for DeepMathQuestion {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathAnswer {
    id: usize,
    question: String,
    model_reasoning: String,
    model_answer: String,
    correct_answer: String,
}

impl HasId for DeepMathAnswer {
    fn id(&self) -> usize {
        self.id
    }
}

async fn evaluate_question(
    question: DeepMathQuestion,
    client: Client,
    model: Model,
) -> DeepMathAnswer {
    let prompt = format!(
        "Please answer the following question by first reasoning and then putting the final short answer in <answer></answer> tags. Question: {}",
        question.question
    );
    let (url, model_name) = match model {
        Model::Gpt5Mini => ("https://api.openai.com/v1/chat/completions", "gpt-5-mini"),
        Model::Qwen25Instruct => (
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
        Model::Qwen25Instruct => client.post(url).json(&body).send().await.unwrap(),
        Model::Gpt5Mini => {
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
    // use regex to extract the answer in <answer></answer> tags
    let re = regex::Regex::new(r"<answer>(.*?)</answer>").unwrap();
    let model_answer = if let Some(caps) = re.captures(&model_reasoning) {
        caps.get(1).map_or("", |m| m.as_str()).to_string()
    } else {
        "No answer found".to_string()
    };
    // Placeholder for actual evaluation logic
    // For demonstration, we just return the question and the final answer
    DeepMathAnswer {
        id: question.id,
        question: question.question,
        model_reasoning,
        model_answer,
        correct_answer: question.final_answer,
    }
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    // load env from .env file
    dotenvy::dotenv().ok();
    let Args { model, num_samples } = Args::parse();
    let model_name = match model {
        Model::Gpt5Mini => "gpt",
        Model::Qwen25Instruct => "qwen",
    };
    let dataset_path = Path::new(INPUT_FOLDER).join(format!("{}.jsonl", num_samples));
    let output_folder = Path::new(OUTPUT_FOLDER_PREFIX).join(format!("{}", model_name));
    let output_file_path = output_folder.join(format!("{}.jsonl", num_samples));
    let client = Client::new();

    let results = parallel_process_jsonl(
        dataset_path,
        output_file_path,
        move |question: DeepMathQuestion| {
            let client = client.clone();
            async move { evaluate_question(question, client, model).await }
        },
        200,
    )
    .await
    .unwrap();

    println!("Processed {} answers", results.len());
}
