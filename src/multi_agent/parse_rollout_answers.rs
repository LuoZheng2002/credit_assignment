use crate::deepmath::generate_raw_answers::Model;
use crate::deepmath::parse_answers::{AnswerParsed, get_parsed_path};
use crate::multi_agent::generate_rollout_answers::{
    RolloutTrajectory, get_rollout_trajectory_path,
};
use crate::parallel_process_jsonl::parallel_process_jsonl;

pub fn extract_boxed_content(text: &str) -> Option<String> {
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

async fn parse_rollout_answer_task(answer: RolloutTrajectory) -> AnswerParsed {
    AnswerParsed {
        id: answer.id,
        model_answer: answer.model_answer,
        correct_answer: answer.correct_answer,
        question: answer.question,
    }
}

pub async fn parse_rollout_answers(model: Model, dataset_name: &str, num_samples: usize) {
    println!(
        "Parsing rollout answers for model {} on {} dataset with {} samples",
        model.cli_name(),
        dataset_name,
        num_samples
    );
    let raw_path = get_rollout_trajectory_path(model, dataset_name, num_samples);
    let parsed_path = get_parsed_path(model, dataset_name, num_samples, true);
    parallel_process_jsonl(
        &[&raw_path],
        &parsed_path,
        |values| {
            assert_eq!(values.len(), 1);
            let answer: RolloutTrajectory =
                serde_json::from_value(values[0].clone()).expect("Failed to parse answer");
            answer
        },
        move |answer: RolloutTrajectory| async move { parse_rollout_answer_task(answer).await },
        2000,
    )
    .await
    .unwrap();
    println!(
        "Parsed rollout answers for model {} on {} dataset with {} samples and saved to {}",
        model.cli_name(),
        dataset_name,
        num_samples,
        parsed_path
    );
}
