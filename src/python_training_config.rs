use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PythonTrainingConfig {
    pub model_cli_name: String,
    pub config_nickname: String,
    pub hyperparameters: TrainingHyperparameters,
    pub num_iterations_limit: usize,
    pub training_trajectory_len_cutoff: usize,
    pub training_mode: TrainingMode,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrainingMode {
    Orchestration {
        epoch: usize,
        training_time: f32,
        input_model_parent_dir: String,
        output_model_parent_dir: String,
        training_summary_dir: String,
    },
    #[serde(rename = "oneshot")]
    OneShot {
        per_epoch_training_time: f32,
        num_oneshot_epochs: usize,
        model_output_root: String,
        training_summary_dir: String,
        base_model_parent_dir: String,
    },
}

impl PythonTrainingConfig {
    pub fn to_json_stdin_payload(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self)
            .map_err(|err| format!("failed to serialize training config as JSON: {}", err))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainingHyperparameters {
    pub lora_or_full: String,
    pub distributed_strategy: String,
    pub advantage_clip: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub grad_accum_steps: usize,
    pub log_time_interval: f32,
    pub seed: u64,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub lr_warmup_steps: usize,
    // lora specific
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_alpha: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_dropout: Option<f32>,
}
