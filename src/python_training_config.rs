use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PythonTrainingConfig {
    #[serde(flatten)]
    pub common: PythonTrainingConfigCommon,
    pub training_time: f32,
    // paths
    pub model_parent_dir: String,
    pub training_trajectory_sqlite_path: String,
    pub checkpoints_parent_dir: String,
    pub final_model_output_parent_dir: String, // final model writes to model/ under this folder
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
