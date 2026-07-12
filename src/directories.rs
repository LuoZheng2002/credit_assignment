use crate::hybrid_dataset::DatasetSplit;

/// Canonical path to the posterior hyperparameters JSON file.
pub const POSTERIOR_HYPERPARAMETERS_PATH: &str = "config/posterior_hyperparameters.json";

/// Canonical path to the validation rollout config JSON file.
pub const VALIDATION_ROLLOUT_CONFIG_PATH: &str = "config/rollout_config_validation_tool.json";

pub fn action_logs_path<S: DatasetSplit>(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let postfix = match S::dataset_file_postfix().as_str() {
        "train" => "training",
        "val" => "validation",
        "test" => "testing",
        other => return Err(format!("Unsupported dataset split postfix: {}", other)),
    };
    Ok(format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/action_logs_{postfix}.extsort"
    ))
}

pub fn inference_wrapper_log_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}/inference_wrapper.txt")
}

pub fn model_parent_dir(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    if epoch == 0 {
        format!("{mount_dir}/large_files/{model_cli_name}")
    } else {
        format!("{mount_dir}/large_files/{model_cli_name}/{config_nickname}/epoch_{epoch}")
    }
}

pub fn model_checkpoint_dir(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    format!("{mount_dir}/large_files/{model_cli_name}/{config_nickname}/epoch_{epoch}")
}

pub fn model_metrics_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    format!(
        "{mount_dir}/small_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/train_metrics.jsonl"
    )
}

pub fn progress_save_path(mount_dir: &str, model_cli_name: &str, config_nickname: &str) -> String {
    format!(
        "{mount_dir}/small_files/{model_cli_name}/{config_nickname}/orchestration_progress.json"
    )
}

pub fn test_accuracy_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    format!(
        "{mount_dir}/small_files/{model_cli_name}/{config_nickname}/test_accuracy_epoch_{epoch}.json"
    )
}

pub fn training_trajectories_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/training_trajectories"
    )
}

pub fn training_trajectories_stats_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/training_trajectories_stats.json"
    )
}

pub fn training_summary_parent_dir(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}/epoch_{epoch}")
}

pub fn training_wrapper_log_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}/training_wrapper.txt")
}

pub fn tui_log_path(mount_dir: &str, model_cli_name: &str, config_nickname: &str) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}/tui_log.bin")
}

pub fn text_logger_summary_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}/text_log_summary.txt")
}

pub fn text_logger_verbose_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}/text_log_verbose.txt")
}

// ---- One-shot path functions ----

pub fn action_logs_oneshot_path<S: DatasetSplit>(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    match S::dataset_file_postfix().as_str() {
        "train" => Ok(format!(
            "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/action_logs_training_oneshot.extsort"
        )),
        "val" => Ok(format!(
            "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/action_logs_validation_oneshot.extsort"
        )),
        "test" => Err("Testing split is not supported for one-shot action logs".to_string()),
        other => Err(format!("Unsupported dataset split postfix: {}", other)),
    }
}

pub fn training_trajectories_oneshot_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!("{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/training_trajectories")
}

pub fn training_trajectories_stats_oneshot_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/training_trajectories_stats.json"
    )
}

pub fn training_summary_oneshot_parent_dir(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}")
}

pub fn rollout_summary_oneshot_path(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
) -> String {
    format!("{mount_dir}/small_files/{model_cli_name}/{config_nickname}/rollout_summary.json")
}

pub fn oneshot_model_parent_dir(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    oneshot_epoch: usize,
) -> String {
    if oneshot_epoch == 0 {
        format!("{mount_dir}/large_files/{model_cli_name}")
    } else {
        format!(
            "{mount_dir}/large_files/{model_cli_name}/{config_nickname}/oneshot_epoch_{oneshot_epoch}"
        )
    }
}

pub fn oneshot_model_checkpoint_dir(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    oneshot_epoch: usize,
) -> String {
    format!(
        "{mount_dir}/large_files/{model_cli_name}/{config_nickname}/oneshot_epoch_{oneshot_epoch}"
    )
}
