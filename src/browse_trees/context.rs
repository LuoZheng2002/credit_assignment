use std::path::Path;

use clap::ValueEnum;

use crate::hybrid_dataset::DatasetSplitEnum;
use crate::llm_model::LlmModelName;

#[derive(Debug, Clone)]
pub(super) struct ActionLogsContext {
    pub model: LlmModelName,
    pub dataset_split: DatasetSplitEnum,
    pub config_nickname: String,
    pub epoch: usize,
}

pub(super) fn parse_action_logs_context(
    action_logs_path: impl AsRef<Path>,
) -> Result<ActionLogsContext, String> {
    let action_logs_path = action_logs_path.as_ref();
    let file_name = action_logs_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Action logs path {} must end with a UTF-8 file or directory name",
                action_logs_path.display()
            )
        })?;
    let dataset_split_name = file_name.strip_suffix(".extsort").unwrap_or(file_name);
    let dataset_split = match dataset_split_name {
        "action_logs_training" => DatasetSplitEnum::Training,
        "action_logs_validation" => DatasetSplitEnum::Validation,
        "action_logs_testing" => DatasetSplitEnum::Testing,
        other => {
            return Err(format!(
                "Unsupported action logs file name {}. Expected action_logs_training[.extsort], action_logs_validation[.extsort], or action_logs_testing[.extsort]",
                other
            ));
        }
    };

    let epoch_dir = action_logs_path.parent().ok_or_else(|| {
        format!(
            "Action logs path {} must have an epoch directory parent",
            action_logs_path.display()
        )
    })?;
    let epoch_dir_name = epoch_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Action logs path {} must have a UTF-8 epoch directory name",
                action_logs_path.display()
            )
        })?;
    let epoch = epoch_dir_name
        .strip_prefix("epoch_")
        .ok_or_else(|| {
            format!(
                "Action logs path {} must be inside an epoch_<n> directory",
                action_logs_path.display()
            )
        })?
        .parse::<usize>()
        .map_err(|err| format!("Failed to parse epoch from {}: {}", epoch_dir_name, err))?;

    let config_dir = epoch_dir.parent().ok_or_else(|| {
        format!(
            "Action logs path {} must be inside a config nickname directory",
            action_logs_path.display()
        )
    })?;
    let config_nickname = config_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Action logs path {} must have a UTF-8 config nickname directory name",
                action_logs_path.display()
            )
        })?
        .to_string();

    let model_dir = config_dir.parent().ok_or_else(|| {
        format!(
            "Action logs path {} must be inside a model directory",
            action_logs_path.display()
        )
    })?;
    let model_cli_name = model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Action logs path {} must have a UTF-8 model directory name",
                action_logs_path.display()
            )
        })?;
    let model = LlmModelName::from_str(model_cli_name, false).map_err(|err| {
        format!(
            "Failed to infer model from action logs path {}: {}",
            action_logs_path.display(),
            err
        )
    })?;

    Ok(ActionLogsContext {
        model,
        dataset_split,
        config_nickname,
        epoch,
    })
}
