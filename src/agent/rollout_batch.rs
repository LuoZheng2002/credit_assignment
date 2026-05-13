use crate::direct_answer::generate_raw_answers::LlmModel;



pub async fn rollout_batch(
    model: LlmModel,
    dataset_name: String,
    num_samples: usize,
    vllm_port: u16,
) {
    
}