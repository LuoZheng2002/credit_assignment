use crate::direct_tool::hybrid_dataset::DatasetSplit;
use crate::utils::mount_dir;

pub fn action_logs_path<S: DatasetSplit>(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    let postfix = match S::dataset_file_postfix().as_str() {
        "train" => "training",
        "val" => "validation",
        "test" => "testing",
        other => return Err(format!("Unsupported dataset split postfix: {}", other)),
    };
    Ok(format!(
        "{md}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/action_logs_{postfix}.extsort"
    ))
}

pub fn inference_wrapper_log_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!("{md}/small_files/{model_cli_name}/{config_nickname}/inference_wrapper.txt"))
}

pub fn model_parent_dir(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    if epoch == 0 {
        Ok(format!("{md}/large_files/{model_cli_name}"))
    } else {
        Ok(format!("{md}/large_files/{model_cli_name}/{config_nickname}/epoch_{epoch}"))
    }
}

pub fn model_checkpoint_dir(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!("{md}/large_files/{model_cli_name}/{config_nickname}/epoch_{epoch}"))
}

pub fn model_metrics_path(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/small_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/train_metrics.jsonl"
    ))
}

pub fn progress_save_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/small_files/{model_cli_name}/{config_nickname}/orchestration_progress.json"
    ))
}

pub fn test_accuracy_path(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/small_files/{model_cli_name}/{config_nickname}/test_accuracy_epoch_{epoch}.json"
    ))
}

pub fn training_trajectories_path(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/training_trajectories"
    ))
}

pub fn training_trajectories_stats_path(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/training_trajectories_stats.json"
    ))
}

pub fn training_summary_parent_dir(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/small_files/{model_cli_name}/{config_nickname}/epoch_{epoch}"
    ))
}

pub fn training_wrapper_log_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!("{md}/small_files/{model_cli_name}/{config_nickname}/training_wrapper.txt"))
}

pub fn tui_log_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!("{md}/small_files/{model_cli_name}/{config_nickname}/tui_log.bin"))
}

// ---- One-shot path functions ----

pub fn action_logs_oneshot_path<S: DatasetSplit>(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    match S::dataset_file_postfix().as_str() {
        "train" => Ok(format!(
            "{md}/medium_files/{model_cli_name}/{config_nickname}/action_logs_training_oneshot.extsort"
        )),
        "val" => Ok(format!(
            "{md}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/action_logs_validation_oneshot.extsort"
        )),
        "test" => Err("Testing split is not supported for one-shot action logs".to_string()),
        other => Err(format!("Unsupported dataset split postfix: {}", other)),
    }
}

pub fn training_trajectories_oneshot_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/medium_files/{model_cli_name}/{config_nickname}/training_trajectories"
    ))
}

pub fn training_trajectories_stats_oneshot_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/medium_files/{model_cli_name}/{config_nickname}/training_trajectories_stats.json"
    ))
}

pub fn training_trajectories_oneshot_path_with_mount(
    model_cli_name: &str,
    config_nickname: &str,
    mount_dir: &str,
) -> Result<String, String> {
    Ok(format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/training_trajectories"
    ))
}

pub fn training_trajectories_stats_oneshot_path_with_mount(
    model_cli_name: &str,
    config_nickname: &str,
    mount_dir: &str,
) -> Result<String, String> {
    Ok(format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/training_trajectories_stats.json"
    ))
}

pub fn training_summary_oneshot_parent_dir(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!("{md}/small_files/{model_cli_name}/{config_nickname}"))
}

pub fn rollout_summary_oneshot_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/small_files/{model_cli_name}/{config_nickname}/rollout_summary.json"
    ))
}

pub fn oneshot_model_parent_dir(
    model_cli_name: &str,
    config_nickname: &str,
    oneshot_epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    if oneshot_epoch == 0 {
        Ok(format!("{md}/large_files/{model_cli_name}"))
    } else {
        Ok(format!(
            "{md}/large_files/{model_cli_name}/{config_nickname}/oneshot_epoch_{oneshot_epoch}"
        ))
    }
}

pub fn oneshot_model_checkpoint_dir(
    model_cli_name: &str,
    config_nickname: &str,
    oneshot_epoch: usize,
) -> Result<String, String> {
    let md = mount_dir()?;
    Ok(format!(
        "{md}/large_files/{model_cli_name}/{config_nickname}/oneshot_epoch_{oneshot_epoch}"
    ))
}
