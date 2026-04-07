use crate::{
    datasets::{DeepMathQuestion, get_question_path},
    deepmath::generate_raw_answers::Model,
    multi_agent::{rollout::rollout, session::SessionLog},
    parallel_process_jsonl::{HasId, parallel_process_jsonl},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct RolloutAnswerRaw {
    pub id: usize,
    pub model_answer: String,
    pub correct_answer: String,
    pub trajectory: SessionLog,
    pub question: String,
}

impl HasId for RolloutAnswerRaw {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_rollout_answer_path(model_name: &str, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/rollout/{}_raw_{}.jsonl",
        model_name, dataset_name, num_samples
    )
}

async fn generate_rollout_answer_task(
    question: DeepMathQuestion,
    client: Client,
    model: Model,
    verifier_probability: f32,
    mut rng: impl rand::Rng,
) -> RolloutAnswerRaw {
    let model_name = model.full_name();
    let session = rollout(
        question.id,
        question.question.clone(),
        client,
        &model_name,
        verifier_probability,
        &mut rng,
    )
    .await;
    RolloutAnswerRaw {
        id: question.id,
        question: question.question,
        model_answer: session
            .session_state
            .final_answer
            .clone()
            .unwrap_or("No answer found".into()),
        correct_answer: question.final_answer,
        trajectory: session.session_log,
    }
}

pub async fn generate_rollout_answers(
    dataset_name: &str,
    num_samples: usize,
    client: Client,
    model: Model,
    verifier_probability: f32,
    rng: &mut StdRng,
) {
    let questions_path = get_question_path(dataset_name, num_samples);
    let raw_output_path = get_rollout_answer_path(model.name(), dataset_name, num_samples);
    let random_seed_u64 = rng.next_u64();
    parallel_process_jsonl(
        &[&questions_path],
        &raw_output_path,
        |values| {
            let question = serde_json::from_value::<DeepMathQuestion>(values[0].clone())
                .expect("Failed to parse question");
            question
        },
        move |question| {
            let new_rng = StdRng::seed_from_u64(random_seed_u64 + question.id() as u64);
            let client = client.clone();
            async move {
                generate_rollout_answer_task(question, client, model, verifier_probability, new_rng)
                    .await
            }
        },
        2000,
    )
    .await
    .unwrap();
}
