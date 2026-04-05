use std::{
    io::BufRead, panic, sync::{Arc, LazyLock}
};

use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::OpenOptions,
    io::AsyncWriteExt,
    sync::{Semaphore, mpsc},
};

#[derive(Parser, Debug)]
#[command(name = "Evaluate DeepMath Model")]
struct Args {
    #[arg(
        short,
        long,
        default_value = "datasets/deepmath_samples/DeepMath-103K-first-20-question_answer.jsonl"
    )]
    input_file: String,
    #[arg(short, long, default_value = "output_deepmath.jsonl")]
    output_file: String,
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
    question: String,
    final_answer: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeepMathAnswer {
    question: String,
    model_answer: String,
    correct_answer: String,
}

pub fn read_jsonl_file(file_path: &str) -> Result<Vec<DeepMathQuestion>, String> {
    let file = std::fs::File::open(file_path).map_err(|e| e.to_string())?;
    let reader = std::io::BufReader::new(file);
    let mut questions = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let question: DeepMathQuestion = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        questions.push(question);
    }
    Ok(questions)
}

// static MODEL: LazyLock<Model> = LazyLock::new(|| {
//     let args = Args::parse();
//     args.model
// });

async fn evaluate_question(
    question: DeepMathQuestion,
    client: Client,
    sem: Arc<Semaphore>,
    model: Model,
) -> DeepMathAnswer {
    let _permit = sem.acquire().await.unwrap(); // Acquire a permit to limit concurrency
    let (url, model_name) = match model {
        Model::Gpt5Mini => (
            "https://api.openai.com/v1/chat/completions",
            "gpt-5-mini",
        ),
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
                "content": question.question
            }
        ],
        "max_tokens": 2048,
    });
    let response = client.post(url).json(&body).send().await.unwrap();
    let json: serde_json::Value = response.json().await.unwrap();
    let model_answer = json["choices"][0]["message"]["content"]
        .as_str()
        .expect(&format!("model answer is invalid: {:?}", json))
        .to_string();
    // Placeholder for actual evaluation logic
    // For demonstration, we just return the question and the final answer
    DeepMathAnswer {
        question: question.question,
        model_answer,
        correct_answer: question.final_answer,
    }
}

async fn receive_lines_and_write_to_file(mut rx: mpsc::Receiver<Vec<u8>>, file_path: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)
        .await
        .unwrap();
    while let Some(line) = rx.recv().await {
        file.write_all(&line).await.unwrap();
        file.write_all(b"\n").await.unwrap();
    }
    file.flush().await.unwrap();
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        std::process::abort();
    }));
    // load env from .env file
    dotenvy::dotenv().ok();
    let args = Args::parse();
    let model = args.model;
    let questions =
        read_jsonl_file("datasets/deepmath_samples/DeepMath-103K-first-20-question_answer.jsonl")
            .unwrap();
    println!("Read {} questions", questions.len());
    let semaphore = Arc::new(Semaphore::new(200)); // Limit to 200 concurrent tasks
    let client = Client::new();

    let (tx, rx) = mpsc::channel::<Vec<u8>>(100);
    let writer_handle = tokio::spawn(receive_lines_and_write_to_file(rx, "output_deepmath.jsonl"));
    for question in questions.into_iter() {
        let owned_permit = semaphore.clone().acquire_owned().await.unwrap(); // Acquire a permit for each task
        let client = client.clone();
        let sem = semaphore.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let answer = evaluate_question(question, client, sem, model).await;
            println!(
                "Question: {}\nModel Answer: {}\nCorrect Answer: {}\n",
                answer.question, answer.model_answer, answer.correct_answer
            );
            let bytes = serde_json::to_vec(&answer).unwrap();
            tx.send(bytes).await.unwrap();
            drop(owned_permit); // Release the permit when done
        });
    }
    drop(tx); // Close the sender to signal completion
    writer_handle.await.unwrap(); // Wait for the writer task to finish
}
