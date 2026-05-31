use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, sleep};

use crate::{
    direct_tool::direct_trajectory::FinalAnswer,
    llm_model::{Gpt4o, Gpt4oLlmCallable, LlmCallable, LlmModelMarker, MyTokenizer},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessJudgment {
    pub model_answer: FinalAnswer,
    pub correct_answer: String,
    pub is_correct: bool,
}

async fn judge_answer_task(
    model_answer: String,
    correct_answer: String,
    question: String,
    client: Client,
) -> bool {
    let prompt = format!(
        "You are an answer checker that checks a model's answer against the reference answer. Judge if the model's answer is equivalent to the reference answer. \
If the model's answer contains units but the reference answer does not, treat them as equivalent if the numerical values are the same. \n\
The question is: \"{}\". \
The model's answer is: \"{}\", and the correct answer is: \"{}\". Return only 'correct' or 'incorrect'.",
        question, model_answer, correct_answer
    );
    let gpt_callable = Gpt4oLlmCallable::new(client);
    let mut last_error: Option<String> = None;

    for attempt in 1..=3 {
        match gpt_callable
            .generate_tokens(
                <Gpt4o as LlmModelMarker>::Tokenizer::tokenize(prompt.clone()).tokens,
                false,
            )
            .await
        {
            Ok(evaluation_tokens) => {
                let evaluation =
                    <Gpt4o as LlmModelMarker>::Tokenizer::decode_i32_ids(&evaluation_tokens)
                        .trim()
                        .to_lowercase();
                if evaluation.contains("incorrect") {
                    return false;
                }
                if evaluation.contains("correct") {
                    return true;
                }
                last_error = Some(format!("Unexpected evaluation result: {}", evaluation));
            }
            Err(error) => {
                last_error = Some(error.to_string());
            }
        }

        if attempt < 3 {
            sleep(Duration::from_secs(1)).await;
        }
    }

    panic!(
        "Failed to judge answer after 3 attempts: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    );
}

pub async fn judge_final_answer(
    final_answer: &FinalAnswer,
    correct_answer: &str,
    question: &str,
    client: Client,
) -> CorrectnessJudgment {
    let is_correct = match final_answer {
        FinalAnswer::ModelProvided(model_answer) => {
            judge_answer_task(
                model_answer.clone(),
                correct_answer.to_string(),
                question.to_string(),
                client,
            )
            .await
        }
        FinalAnswer::Failure(_error_message) => false,
    };
    CorrectnessJudgment {
        model_answer: final_answer.clone(),
        correct_answer: correct_answer.to_string(),
        is_correct,
    }
}
