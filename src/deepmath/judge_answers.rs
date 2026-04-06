use std::fs::File;

use indexmap::IndexMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    deepmath::{
        generate_raw_answers::Model,
        parse_answers::{DeepMathAnswerParsed, get_deepmath_parsed_path},
    },
    parallel_process_jsonl::{HasId, parallel_process_jsonl, read_json_lines_indexed},
};

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathCorrectness {
    pub id: usize,
    pub correct: bool,
    pub model_answer: String,
    pub correct_answer: String,
    pub question: String,
}

impl HasId for DeepMathCorrectness {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_deepmath_correctness_path(model_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/deepmath_correctness_{}.jsonl",
        model_name, num_samples
    )
}

fn get_deepmath_accuracy_path(model_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/deepmath_accuracy_{}.json",
        model_name, num_samples
    )
}

async fn judge_answer_task(answer: DeepMathAnswerParsed, client: Client) -> DeepMathCorrectness {
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
    DeepMathCorrectness {
        id: answer.id,
        correct,
        model_answer: answer.model_answer,
        correct_answer: answer.correct_answer,
        question: answer.question,
    }
}

pub async fn judge_answers(model: Model, num_samples: usize, client: Client) {
    println!(
        "Judging answers for model {} on {} samples",
        model.name(),
        num_samples
    );
    let model_name = model.name();
    let parsed_answers_path = get_deepmath_parsed_path(model_name, num_samples);
    let correctness_path = get_deepmath_correctness_path(model_name, num_samples);
    parallel_process_jsonl(
        &[&parsed_answers_path],
        &correctness_path,
        |values| {
            assert_eq!(values.len(), 1);
            let answer: DeepMathAnswerParsed =
                serde_json::from_value(values[0].clone()).expect("Failed to parse answer");
            answer
        },
        move |answer: DeepMathAnswerParsed| {
            let client = client.clone();
            async move { judge_answer_task(answer, client).await }
        },
        2000,
    )
    .await
    .unwrap();
    println!(
        "Finished judging answers for model {} on {} samples. Results saved to {}",
        model.name(),
        num_samples,
        correctness_path
    );
    let score_results: IndexMap<usize, DeepMathCorrectness> =
        read_json_lines_indexed(&File::open(&correctness_path).unwrap()).unwrap();
    let mut correct = 0;
    for score in score_results.values() {
        if score.correct {
            correct += 1;
        }
    }
    let score_stats_file = get_deepmath_accuracy_path(model_name, num_samples);
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
        "Accuracy for model {} on {} samples: {:.2}%",
        model.name(),
        num_samples,
        correct as f64 / score_results.len() as f64 * 100.0
    );
}
