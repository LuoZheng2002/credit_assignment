use crate::deepmath::generate_raw_answers::{DeepMathAnswerRaw, get_raw_answer_path};
use crate::parallel_process_jsonl::{HasId, parallel_process_jsonl};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct DeepMathAnswerParsed {
    pub id: usize,
    pub model_answer: String,
    pub correct_answer: String,
    pub question: String,
}

impl HasId for DeepMathAnswerParsed {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_parsed_path(model_name: &str, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/{}_parsed_{}.jsonl",
        model_name, dataset_name, num_samples
    )
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

async fn parse_model_answer_task(answer: DeepMathAnswerRaw) -> DeepMathAnswerParsed {
    let model_answer = extract_boxed_content(&answer.model_reasoning)
        .unwrap_or_else(|| "No answer found".to_string());
    DeepMathAnswerParsed {
        id: answer.id,
        model_answer,
        correct_answer: answer.correct_answer,
        question: answer.question,
    }
}

pub async fn parse_answers(model_name: &str, dataset_name: &str, num_samples: usize) {
    println!(
        "Parsing model answers for model {} on {} dataset with {} samples",
        model_name, dataset_name, num_samples
    );
    let raw_path = get_raw_answer_path(model_name, dataset_name, num_samples);
    let parsed_path = get_parsed_path(model_name, dataset_name, num_samples);
    parallel_process_jsonl(
        &[&raw_path],
        &parsed_path,
        |values| {
            assert_eq!(values.len(), 1);
            let answer: DeepMathAnswerRaw =
                serde_json::from_value(values[0].clone()).expect("Failed to parse answer");
            answer
        },
        move |answer: DeepMathAnswerRaw| async move { parse_model_answer_task(answer).await },
        2000,
    )
    .await
    .unwrap();
    println!(
        "Parsed model answers for model {} on {} dataset with {} samples and saved to {}",
        model_name, dataset_name, num_samples, parsed_path
    );
}
