use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::utils::storage_medium_files_dir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPaths {
    pub training_rollout_config_path: String,
    pub validation_rollout_config_path: String,
    pub testing_rollout_config_path: String,
    pub posterior_hyperparameters_path: String,
}

pub fn config_paths_file_path(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<PathBuf, String> {
    Ok(PathBuf::from(storage_medium_files_dir()?)
        .join(model_cli_name)
        .join(config_nickname)
        .join("config_paths.json"))
}

pub fn config_paths_file_path_from_action_logs_path(
    action_logs_path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let action_logs_path = action_logs_path.as_ref();
    let config_paths_dir = action_logs_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            format!(
                "Action logs path {} must look like .../epoch_<n>/action_logs_<split>.sqlite",
                action_logs_path.display()
            )
        })?;
    Ok(config_paths_dir.join("config_paths.json"))
}

pub fn derive_testing_rollout_config_path(
    validation_rollout_config_path: impl AsRef<Path>,
) -> Result<String, String> {
    let validation_rollout_config_path = validation_rollout_config_path.as_ref();
    let file_name = validation_rollout_config_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Cannot derive testing rollout config path from {} because it has no UTF-8 file name",
                validation_rollout_config_path.display()
            )
        })?;

    if !file_name.contains("validation") {
        return Err(format!(
            "Cannot derive testing rollout config path from {} because the file name does not contain 'validation'",
            validation_rollout_config_path.display()
        ));
    }

    let testing_file_name = file_name.replacen("validation", "testing", 1);
    Ok(validation_rollout_config_path
        .with_file_name(testing_file_name)
        .display()
        .to_string())
}
