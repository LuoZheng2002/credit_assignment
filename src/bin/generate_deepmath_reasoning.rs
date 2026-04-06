use clap::Parser;
use credit_assignment::parallel_process_jsonl::{HasId, parallel_process_jsonl};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(short, long)]
    num_samples: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepMathQuestionReasoning {
    id: usize,
    reasoning: String,
    final_answer: String,
    question: String,
}

impl HasId for DeepMathQuestionReasoning {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepMathConciseReasoning {
    id: usize,
    concise_reasoning: String,
    final_answer: String,
    question: String,
}

impl HasId for DeepMathConciseReasoning {
    fn id(&self) -> usize {
        self.id
    }
}

async fn generate_reasoning(
    question: DeepMathQuestionReasoning,
    client: Client,
) -> DeepMathConciseReasoning {
    let prompt = format!(
        "You are a helpful assistant that provides concise reasoning steps for solving math problems. \
Given the following question, final answer and a reference reasoning paragraph, generate a concise reasoning process that leads to the final answer, with only the broad ideas and key intermediate results, without detailed calculations. \n\
Question: {}\nFinal Answer: {}\nReasoning: {}\nConcise Reasoning:",
        question.question, question.final_answer, question.reasoning
    );
    let body = json!({
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

fn get_questions_with_reasoning_path(num_samples: usize) -> String {
    format!("datasets/deepmath_samples_{}_reasoning.jsonl", num_samples)
}

fn get_concise_reasoning_output_path(num_samples: usize) -> String {
    format!(
        "datasets/deepmath_samples_{}_concise_reasoning.jsonl",
        num_samples
    )
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    let Args { num_samples } = Args::parse();
    let questions_with_reasoning_path = get_questions_with_reasoning_path(num_samples);
    let concise_reasoning_output_path = get_concise_reasoning_output_path(num_samples);
    let client = Client::new();
    parallel_process_jsonl(
        &questions_with_reasoning_path,
        &concise_reasoning_output_path,
        move |question| {
            let client = client.clone();
            async move { generate_reasoning(question, client).await }
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
