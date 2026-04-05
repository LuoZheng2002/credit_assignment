use std::{
    io::BufRead,
    panic,
    sync::{Arc, LazyLock, atomic::AtomicUsize},
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
    model_reasoning: String,
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
        question: question.question,
        model_reasoning,
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
    let count: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    for question in questions.into_iter() {
        let owned_permit = semaphore.clone().acquire_owned().await.unwrap(); // Acquire a permit for each task
        let client = client.clone();
        let sem = semaphore.clone();
        let tx = tx.clone();
        let count = count.clone();
        tokio::spawn(async move {
            let answer = evaluate_question(question, client, sem, model).await;
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            println!(
                "Processed {} questions",
                count.load(std::sync::atomic::Ordering::SeqCst)
            );
            let bytes = serde_json::to_vec(&answer).unwrap();
            tx.send(bytes).await.unwrap();
            drop(owned_permit); // Release the permit when done
        });
    }
    drop(tx); // Close the sender to signal completion
    writer_handle.await.unwrap(); // Wait for the writer task to finish
}
