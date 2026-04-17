use indexmap::IndexMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    call_llm::call_llm_chat_completions,
    deepmath::{
        generate_raw_answers::Model,
        parse_answers::{AnswerParsed, get_parsed_path},
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

pub fn get_correctness_path(
    model: Model,
    dataset_name: &str,
    num_samples: usize,
    is_rollout: bool,
) -> String {
    if is_rollout {
        format!(
            "results/{}/rollout/{}_correctness_{}.jsonl",
            model.cli_name(),
            dataset_name,
            num_samples
        )
    } else {
        format!(
            "results/{}/{}_correctness_{}.jsonl",
            model.cli_name(),
            dataset_name,
            num_samples
        )
    }
}

pub fn get_accuracy_path(
    model: Model,
    dataset_name: &str,
    num_samples: usize,
    is_rollout: bool,
) -> String {
    if is_rollout {
        format!(
            "results/{}/rollout/{}_accuracy_{}.json",
            model.cli_name(),
            dataset_name,
            num_samples
        )
    } else {
        format!(
            "results/{}/{}_accuracy_{}.json",
            model.cli_name(),
            dataset_name,
            num_samples
        )
    }
}

pub async fn judge_answer_task(answer: AnswerParsed, client: Client) -> DeepMathCorrectness {
    let prompt = format!(
        // "The question is: {}. The model's answer is: {}. The correct answer is: {}. Please evaluate whether the model's answer is correct and return only 'correct' or 'incorrect'.",
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
The question is: \"{}\". \
The model's answer is: \"{}\", and the correct answer is: \"{}\". Return only 'correct' or 'incorrect'.",
        answer.question, answer.model_answer, answer.correct_answer
    );
    let evaluation = call_llm_chat_completions(client, prompt, Model::Gpt4o, false)
        .await
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

pub async fn judge_answers(
    model: Model,
    dataset_name: &str,
    num_samples: usize,
    client: Client,
    is_rollout: bool,
) {
    println!(
        "Judging answers for model {} on {} samples",
        model.cli_name(),
        num_samples
    );
    let parsed_answers_path = get_parsed_path(model, dataset_name, num_samples, is_rollout);
    let correctness_path = get_correctness_path(model, dataset_name, num_samples, is_rollout);
    parallel_process_jsonl(
        &[&parsed_answers_path],
        &correctness_path,
        |values| {
            assert_eq!(values.len(), 1);
            let answer: AnswerParsed =
                serde_json::from_value(values[0].clone()).expect("Failed to parse answer");
            answer
        },
        move |answer: AnswerParsed| {
            let client = client.clone();
            async move { judge_answer_task(answer, client).await }
        },
        2000,
    )
    .await
    .unwrap();
    println!(
        "Finished judging answers for model {} on {} samples. Results saved to {}",
        model.cli_name(),
        num_samples,
        correctness_path
    );
    let score_results: IndexMap<usize, DeepMathCorrectness> =
        read_json_lines_indexed(&correctness_path).unwrap();
    let mut correct = 0;
    for score in score_results.values() {
        if score.correct {
            correct += 1;
        }
    }
    let score_stats_file = get_accuracy_path(model, dataset_name, num_samples, is_rollout);
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
        model.cli_name(),
        num_samples,
        correct as f64 / score_results.len() as f64 * 100.0
    );
}
