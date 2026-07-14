use credit_assignment::python_training_config::{
    PythonTrainingConfig, TrainingHyperparameters, TrainingMode,
};
use serde_json::Value;

#[test]
fn training_config_orchestration_serializes_to_json_stdin_payload() {
    let config = PythonTrainingConfig {
        hyperparameters: TrainingHyperparameters {
            lora_or_full: "lora".to_string(),
            distributed_strategy: "ddp".to_string(),
            advantage_clip: 1.5,
            learning_rate: 0.0001,
            weight_decay: 0.01,
            grad_accum_steps: 8,
            log_time_interval: 5.0,
            seed: 7,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            lr_warmup_steps: 100,
            lora_rank: Some(64),
            lora_alpha: None,
            lora_dropout: Some(0.05),
        },
        num_iterations_limit: 200,
        model_cli_name: "qwen35_4b".to_string(),
        config_nickname: "demo".to_string(),
        training_mode: TrainingMode::Orchestration {
            epoch: 3,
            training_time: 120.0,
            input_model_parent_dir: "/tmp/artifacts/large_files/qwen35_4b/demo/epoch_3".to_string(),
            output_model_parent_dir: "/tmp/artifacts/large_files/qwen35_4b/demo/epoch_4"
                .to_string(),
            training_summary_dir: "/tmp/artifacts/small_files/qwen35_4b/demo/epoch_3".to_string(),
        },
    };

    let payload = config
        .to_json_stdin_payload()
        .expect("config should serialize into JSON stdin payload");
    let parsed: Value =
        serde_json::from_slice(&payload).expect("stdin payload should parse as JSON object");

    let hp = &parsed["hyperparameters"];
    assert_eq!(hp["lora_or_full"], "lora");
    assert_eq!(hp["distributed_strategy"], "ddp");
    assert_eq!(hp["lora_rank"], 64);
    let adam_beta2 = hp["adam_beta2"]
        .as_f64()
        .expect("expected adam_beta2 to deserialize as f64");
    assert!((adam_beta2 - 0.95).abs() < 0.0001);
    let dropout = hp["lora_dropout"]
        .as_f64()
        .expect("expected lora_dropout to deserialize as f64");
    assert!((dropout - 0.05).abs() < 0.0001);
    assert!(hp.get("lora_alpha").is_none());

    // training_summary_parent_dir no longer exists at top level
    assert!(parsed.get("training_summary_parent_dir").is_none());
    let mode = &parsed["training_mode"];
    assert_eq!(mode["type"], "orchestration");
    assert_eq!(mode["epoch"], 3);
    assert_eq!(mode["training_time"], 120.0);
    assert_eq!(
        mode["input_model_parent_dir"],
        "/tmp/artifacts/large_files/qwen35_4b/demo/epoch_3"
    );
    assert_eq!(
        mode["output_model_parent_dir"],
        "/tmp/artifacts/large_files/qwen35_4b/demo/epoch_4"
    );
    assert_eq!(
        mode["training_summary_dir"],
        "/tmp/artifacts/small_files/qwen35_4b/demo/epoch_3"
    );
}

#[test]
fn training_config_oneshot_serializes_to_json_stdin_payload() {
    let config = PythonTrainingConfig {
        hyperparameters: TrainingHyperparameters {
            lora_or_full: "full".to_string(),
            distributed_strategy: "ddp".to_string(),
            advantage_clip: 2.0,
            learning_rate: 0.00005,
            weight_decay: 0.01,
            grad_accum_steps: 16,
            log_time_interval: 10.0,
            seed: 1,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            lr_warmup_steps: 100,
            lora_rank: None,
            lora_alpha: None,
            lora_dropout: None,
        },
        num_iterations_limit: 50,
        model_cli_name: "qwen35_4b".to_string(),
        config_nickname: "oneshot_demo".to_string(),
        training_mode: TrainingMode::OneShot {
            per_epoch_training_time: 30.0,
            num_oneshot_epochs: 3,
            model_output_root: "/tmp/artifacts/large_files/qwen35_4b/oneshot_demo".to_string(),
            training_summary_dir: "/tmp/artifacts/small_files/qwen35_4b/oneshot_demo".to_string(),
            base_model_parent_dir: "/tmp/artifacts/large_files/qwen35_4b".to_string(),
        },
    };

    let payload = config
        .to_json_stdin_payload()
        .expect("config should serialize into JSON stdin payload");
    let parsed: Value =
        serde_json::from_slice(&payload).expect("stdin payload should parse as JSON object");

    let mode = &parsed["training_mode"];
    assert_eq!(mode["type"], "oneshot");
    assert_eq!(mode["per_epoch_training_time"], 30.0);
    assert_eq!(mode["num_oneshot_epochs"], 3);
    assert_eq!(
        mode["model_output_root"],
        "/tmp/artifacts/large_files/qwen35_4b/oneshot_demo"
    );
    assert_eq!(
        mode["training_summary_dir"],
        "/tmp/artifacts/small_files/qwen35_4b/oneshot_demo"
    );
}
