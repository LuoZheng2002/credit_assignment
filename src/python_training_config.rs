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
    pub adam_fp32: bool,
    pub epoch: usize,
    pub model_parent_dir: String,
    pub checkpoints_parent_dir: String,
    pub final_model_output_parent_dir: String,
    pub training_summary_parent_dir: String,
    #[serde(default = "default_training_mode")]
    #[serde(skip_serializing_if = "is_default_training_mode")]
    pub training_mode: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "usize_is_zero")]
    pub oneshot_num_epochs: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "usize_is_zero")]
    pub oneshot_start_epoch: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "String::is_empty")]
    pub oneshot_model_output_root: String,
}

fn default_training_mode() -> String {
    "orchestration".to_string()
}

fn is_default_training_mode(value: &str) -> bool {
    value == "orchestration"
}

fn usize_is_zero(value: &usize) -> bool {
    *value == 0
}

impl PythonTrainingConfig {
    pub fn to_json_stdin_payload(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self)
            .map_err(|err| format!("failed to serialize training config as JSON: {}", err))
    }
}

fn default_adam_beta1() -> f32 {
    0.9
}

fn default_adam_beta2() -> f32 {
    0.95
}

fn default_lr_schedule() -> String {
    "cosine".to_string()
}

fn default_lr_total_steps() -> usize {
    0
}

fn default_kl_and_ema_enabled() -> bool {
    false
}

fn default_kl_penalty_coefficient() -> f32 {
    0.04
}

fn default_ema_decay() -> f32 {
    0.992
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
    #[serde(default = "default_adam_beta1")]
    pub adam_beta1: f32,
    #[serde(default = "default_adam_beta2")]
    pub adam_beta2: f32,
    #[serde(default = "default_lr_schedule")]
    pub lr_schedule: String,
    #[serde(default = "default_lr_total_steps")]
    pub lr_total_steps: usize,
    // lora specific
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_alpha: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_dropout: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_target_modules_csv: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_checkpoint_tag: Option<String>,
    #[serde(default = "default_kl_and_ema_enabled")]
    pub kl_and_ema_enabled: bool,
    #[serde(default = "default_kl_penalty_coefficient")]
    pub kl_penalty_coefficient: f32,
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f32,
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{PythonTrainingConfig, PythonTrainingConfigCommon};

    #[test]
    fn training_config_serializes_to_json_stdin_payload() {
        let config = PythonTrainingConfig {
            common: PythonTrainingConfigCommon {
                training_plan: "lora".to_string(),
                advantage_clip: 1.5,
                learning_rate: 0.0001,
                weight_decay: 0.01,
                grad_accum_steps: 8,
                log_time_interval: 5.0,
                checkpoint_save_time_interval: 60.0,
                seed: 7,
                adam_beta1: 0.9,
                adam_beta2: 0.95,
                lr_schedule: "cosine".to_string(),
                lr_total_steps: 1000,
                lora_rank: Some(64),
                lora_alpha: None,
                lora_dropout: Some(0.05),
                lora_target_modules_csv: None,
                resume_checkpoint_tag: Some("latest".to_string()),
                kl_and_ema_enabled: false,
                kl_penalty_coefficient: 0.04,
                ema_decay: 0.992,
            },
            training_time: 120.0,
            num_iterations_limit: 200,
            artifact_root_dir: "/tmp/artifacts".to_string(),
            hpc_training_root_dir: None,
            model_cli_name: "qwen35_4b".to_string(),
            config_nickname: "demo".to_string(),
            adam_fp32: true,
            epoch: 3,
            model_parent_dir: "/tmp/artifacts/results/qwen35_4b/demo/epoch_3".to_string(),
            checkpoints_parent_dir: "/tmp/artifacts/results/qwen35_4b/demo/epoch_3".to_string(),
            final_model_output_parent_dir: "/tmp/artifacts/results/qwen35_4b/demo/epoch_4"
                .to_string(),
            training_summary_parent_dir: "/tmp/artifacts/results/qwen35_4b/demo/epoch_3"
                .to_string(),
            training_mode: "orchestration".to_string(),
            oneshot_num_epochs: 0,
            oneshot_start_epoch: 0,
            oneshot_model_output_root: String::new(),
        };

        let payload = config
            .to_json_stdin_payload()
            .expect("config should serialize into JSON stdin payload");
        let parsed: Value =
            serde_json::from_slice(&payload).expect("stdin payload should parse as JSON object");

        assert_eq!(parsed["training_plan"], "lora");
        assert_eq!(parsed["artifact_root_dir"], "/tmp/artifacts");
        assert_eq!(parsed["epoch"], 3);
        assert_eq!(parsed["lora_rank"], 64);
        let adam_beta2 = parsed["adam_beta2"]
            .as_f64()
            .expect("expected adam_beta2 to deserialize as f64");
        assert!((adam_beta2 - 0.95).abs() < 0.0001);
        let dropout = parsed["lora_dropout"]
            .as_f64()
            .expect("expected lora_dropout to deserialize as f64");
        assert!((dropout - 0.05).abs() < 0.0001);
        assert_eq!(parsed["resume_checkpoint_tag"], "latest");
        assert!(parsed.get("hpc_training_root_dir").is_none());
        assert!(parsed.get("lora_alpha").is_none());
    }
}
