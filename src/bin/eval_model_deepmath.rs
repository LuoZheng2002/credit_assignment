use std::fs::File;

use clap::Parser;
use credit_assignment::parallel_process_jsonl::{
    HasId, parallel_process_jsonl, read_json_lines_indexed,
};
use indexmap::IndexMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};

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
    correct_answer: String,
    model_reasoning: String,
    question: String,
}

impl HasId for DeepMathAnswer {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathAnswerParsed {
    id: usize,
    model_answer: String,
    correct_answer: String,
    question: String,
}

impl HasId for DeepMathAnswerParsed {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathScore {
    id: usize,
    correct: bool,
    model_answer: String,
    correct_answer: String,
    question: String,
}

impl HasId for DeepMathScore {
    fn id(&self) -> usize {
        self.id
    }
}

async fn parse_model_answer(answer: DeepMathAnswer) -> DeepMathAnswerParsed {
    let model_answer = extract_boxed_content(&answer.model_reasoning)
        .unwrap_or_else(|| "No answer found".to_string());
    DeepMathAnswerParsed {
        id: answer.id,
        model_answer,
        correct_answer: answer.correct_answer,
        question: answer.question,
    }
}

async fn evaluate_question(
    question: DeepMathQuestion,
    client: Client,
    model: Model,
) -> DeepMathAnswer {
    let prompt = format!(
        "Please answer the following question by first reasoning and then putting the final short answer in \\boxed{{}}. Question: {}",
        question.question
    );
    let (url, model_name) = match model {
        Model::Gpt5Mini => ("https://api.openai.com/v1/chat/completions", "gpt-4o"),
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
    let mut model_reasoning = json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("model answer is invalid: {:?}", json))
        .to_string();
    // remove all new lines from model_reasoning
    model_reasoning.retain(|c| c != '\n');
    // Placeholder for actual evaluation logic
    // For demonstration, we just return the question and the final answer
    DeepMathAnswer {
        id: question.id,
        question: question.question,
        model_reasoning,
        correct_answer: question.final_answer,
    }
}

fn extract_boxed_content(text: &str) -> Option<String> {
    const MARKER: &str = "\\boxed{";
    let start = text.find(MARKER)?;
    let mut bracket_depth = 1;
    let mut content = String::new();
    for ch in text[start + MARKER.len()..].chars() {
        match ch {
            '{' => {
                bracket_depth += 1;
                content.push(ch);
            }
            '}' => {
                bracket_depth -= 1;
                if bracket_depth == 0 {
                    return Some(content);
                }
                content.push(ch);
            }
            other => content.push(other),
        }
    }
    None
}

async fn score_result(answer: DeepMathAnswerParsed, client: Client) -> DeepMathScore {
    let prompt = format!(
        // "The question is: {}. The model's answer is: {}. The correct answer is: {}. Please evaluate whether the model's answer is correct and return only 'correct' or 'incorrect'.",
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
The model's answer is: \"{}\", and the correct answer is: \"{}\". Return only 'correct' or 'incorrect'.",
        answer.model_answer, answer.correct_answer
    );
    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_completion_tokens": 2048,
    });
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .unwrap();
    let json: serde_json::Value = response.json().await.unwrap();
    let evaluation = json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("evaluation is invalid: {:?}", json))
        .trim()
        .to_lowercase();
    println!("Evaluation for question {}: {}", answer.id, evaluation);
    let correct = match evaluation.as_str() {
        "correct" => true,
        "incorrect" => false,
        _ => {
            println!(
                "Unexpected evaluation result for question {}: {}. Treating it as incorrect.",
                answer.id, evaluation
            );
            false
        }
    };
    DeepMathScore {
        id: answer.id,
        correct,
        model_answer: answer.model_answer,
        correct_answer: answer.correct_answer,
        question: answer.question,
    }
}

fn get_deepmath_dataset_path(num_samples: usize) -> String {
    format!("datasets/deepmath_samples_{}.jsonl", num_samples)
}

fn get_deepmath_raw_output_path(model_name: &str, num_samples: usize) -> String {
    format!("results/{}/deepmath_raw_{}.jsonl", model_name, num_samples)
}

fn get_deepmath_parsed_output_path(model_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/deepmath_parsed_{}.jsonl",
        model_name, num_samples
    )
}

fn get_deepmath_score_output_path(model_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/deepmath_scores_{}.jsonl",
        model_name, num_samples
    )
}
fn get_deepmath_score_stats_output_path(model_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/deepmath_scores_{}_stats.json",
        model_name, num_samples
    )
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
    let client = Client::new();

    let deepmath_dataset_path = get_deepmath_dataset_path(num_samples);
    let deepmath_raw_path = get_deepmath_raw_output_path(model_name, num_samples);
    println!(
        "Evaluating model {} on DeepMath dataset with {} samples",
        model_name, num_samples
    );
    {
        let client = client.clone();
        parallel_process_jsonl(
            &deepmath_dataset_path,
            &deepmath_raw_path,
            move |question: DeepMathQuestion| {
                let client = client.clone();
                async move { evaluate_question(question, client, model).await }
            },
            2000,
        )
        .await
        .unwrap();
    }
    // parse model answers to extract the final answer from the reasoning
    let deepmath_parsed_path = get_deepmath_parsed_output_path(model_name, num_samples);
    println!(
        "Parsing model answers for model {} on DeepMath dataset with {} samples",
        model_name, num_samples
    );
    parallel_process_jsonl(
        &deepmath_raw_path,
        &deepmath_parsed_path,
        move |answer: DeepMathAnswer| async move { parse_model_answer(answer).await },
        2000,
    )
    .await
    .unwrap();

    let deepmath_score_output_path = get_deepmath_score_output_path(model_name, num_samples);
    println!(
        "Scoring model answers for model {} on DeepMath dataset with {} samples",
        model_name, num_samples
    );
    parallel_process_jsonl(
        &deepmath_parsed_path,
        &deepmath_score_output_path,
        move |answer: DeepMathAnswerParsed| {
            let client = client.clone();
            async move { score_result(answer, client).await }
        },
        2000,
    )
    .await
    .unwrap();
    let score_results: IndexMap<usize, DeepMathScore> =
        read_json_lines_indexed(&File::open(&deepmath_score_output_path).unwrap()).unwrap();
    let mut correct = 0;
    for score in score_results.values() {
        if score.correct {
            correct += 1;
        }
    }
    let score_stats_file = get_deepmath_score_stats_output_path(model_name, num_samples);
    std::fs::write(
        score_stats_file,
        serde_json::to_string_pretty(&serde_json::json!({
            "total": score_results.len(),
            "correct": correct,
            "accuracy": correct as f64 / score_results.len() as f64,
        }))
        .unwrap(),
    )
    .unwrap();
    println!(
        "Finished evaluating model {} on DeepMath dataset with {} samples. Accuracy: {:.2}%",
        model_name,
        num_samples,
        correct as f64 / score_results.len() as f64 * 100.0
    );
}
