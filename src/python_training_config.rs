use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PythonTrainingConfig {
    #[serde(flatten)]
    pub common: PythonTrainingConfigCommon,
    pub training_time: f32,
    pub num_iterations_limit: usize,
    pub artifact_root_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hpc_training_root_dir: Option<String>,
    pub model_cli_name: String,
    pub config_nickname: String,
    pub epoch: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PythonTrainingConfigCommon {
    pub training_plan: String,
    pub advantage_clip: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub grad_accum_steps: usize,
    pub log_time_interval: f32,
    pub checkpoint_save_time_interval: f32,
    pub seed: u64,
    // lora specific
    pub lora_rank: Option<usize>,
    pub lora_alpha: Option<usize>,
    pub lora_dropout: Option<f32>,
    pub lora_target_modules_csv: Option<String>,
    pub resume_checkpoint_tag: Option<String>,
}
