use crate::{
    deepmath::generate_raw_answers::Model, multi_agent::session::Tree,
    parallel_process_jsonl::HasId,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StepQualityRatio {
    pub tool_numerator: usize,
    pub tool_denominator: usize,
    pub complete_numerator: usize,
    pub complete_denominator: usize,
    pub focused_numerator: usize,
    pub focused_denominator: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CountRatio {
    pub numerator: usize,
    pub denominator: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RolloutTrajectory {
    pub id: usize,
    pub correct_answer: String,
    pub step_quality_ratio: StepQualityRatio,
    pub failed_and_aborted_ratio: CountRatio,
    pub trajectory: Tree,
    pub question: String,
}

impl HasId for RolloutTrajectory {
    fn id(&self) -> usize {
        self.id
    }
}

pub fn get_rollout_trajectory_path(model: Model, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/rollout/{}_trajectory_{}.jsonl",
        model.cli_name(),
        dataset_name,
        num_samples
    )
}

pub fn get_session_log_path(model: Model, dataset_name: &str, num_samples: usize) -> String {
    format!(
        "results/{}/rollout/{}_session_log_{}.jsonl",
        model.cli_name(),
        dataset_name,
        num_samples
    )
}

// async fn generate_rollout_answer_task(
//     question: DeepMathQuestion,
//     client: Client,
//     model: Model,
//     verifier_probability: f32,
//     mut rng: impl rand::Rng,
// ) -> RolloutTrajectory {
//     let loaded_session_log = todo!();
//     let session = rollout(
//         question.id,
//         question.question.clone(),
//         loaded_session_log,
//         client,
//         model,
//         verifier_probability,
//         &mut rng,
//     )
//     .await;
//     RolloutTrajectory {
//         id: question.id,
//         question: question.question,
//         model_answer: session
//             .session_state
//             .final_answer
//             .clone()
//             .unwrap_or("No answer found".into()),
//         correct_answer: question.final_answer,
//         trajectory: session.session_log,
//     }
// }

// pub async fn generate_rollout_answers(
//     dataset_name: &str,
//     num_samples: usize,
//     client: Client,
//     model: Model,
//     verifier_probability: f32,
//     rng: &mut StdRng,
// ) {
//     let questions_path = get_question_path(dataset_name, num_samples);
//     let raw_output_path = get_rollout_answer_path(model, dataset_name, num_samples);
//     let random_seed_u64 = rng.next_u64();
//     parallel_process_jsonl(
//         &[&questions_path],
//         &raw_output_path,
//         |values| {
//             let question = serde_json::from_value::<DeepMathQuestion>(values[0].clone())
//                 .expect("Failed to parse question");
//             question
//         },
//         move |question| {
//             let new_rng = StdRng::seed_from_u64(random_seed_u64 + question.id() as u64);
//             let client = client.clone();
//             async move {
//                 generate_rollout_answer_task(question, client, model, verifier_probability, new_rng)
//                     .await
//             }
//         },
//         2000,
//     )
//     .await
//     .unwrap();
// }
