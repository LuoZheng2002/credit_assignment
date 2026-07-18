use std::path::{Path, PathBuf};

use research_utility::progress_text_logger::log_info;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct OneshotRunManifest {
    pub num_oneshot_epochs: usize,
}

pub fn oneshot_run_manifest_path(summary_parent_dir: &str) -> PathBuf {
    Path::new(summary_parent_dir).join("oneshot_run_manifest.json")
}

pub fn oneshot_aggregated_summary_path(summary_parent_dir: &str) -> PathBuf {
    Path::new(summary_parent_dir).join("oneshot_aggregated_summary.json")
}

pub fn write_oneshot_run_manifest(summary_parent_dir: &str, num_oneshot_epochs: usize) {
    std::fs::create_dir_all(summary_parent_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create oneshot summary parent dir {}: {}",
            summary_parent_dir, err
        )
    });
    let manifest_path = oneshot_run_manifest_path(summary_parent_dir);
    let payload = OneshotRunManifest { num_oneshot_epochs };
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&payload).unwrap() + "\n",
    )
    .unwrap_or_else(|err| {
        panic!(
            "Failed to write oneshot run manifest to {}: {}",
            manifest_path.display(),
            err
        )
    });
    log_info(format!(
        "Wrote oneshot run manifest to {}",
        manifest_path.display()
    ));
}

pub fn read_oneshot_run_manifest(summary_parent_dir: &str) -> Option<OneshotRunManifest> {
    let manifest_path = oneshot_run_manifest_path(summary_parent_dir);
    if !manifest_path.exists() {
        return None;
    }
    std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|content| serde_json::from_str::<OneshotRunManifest>(&content).ok())
}

pub fn read_oneshot_epoch_count_from_summary(summary_parent_dir: &str) -> Option<usize> {
    let aggregated_path = oneshot_aggregated_summary_path(summary_parent_dir);
    if !aggregated_path.exists() {
        return None;
    }
    std::fs::read_to_string(&aggregated_path)
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .and_then(|parsed| parsed.get("num_oneshot_epochs").and_then(|v| v.as_u64()))
        .and_then(|value| usize::try_from(value).ok())
}

pub fn detect_oneshot_artifacts(summary_parent_dir: &str, model_output_root: &str) -> bool {
    if oneshot_run_manifest_path(summary_parent_dir).exists()
        || oneshot_aggregated_summary_path(summary_parent_dir).exists()
    {
        return true;
    }

    if let Ok(entries) = std::fs::read_dir(summary_parent_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            if file_name
                .to_str()
                .is_some_and(|name| name.starts_with("oneshot_per_epoch_summary_epoch_"))
            {
                return true;
            }
        }
    }

    if let Ok(entries) = std::fs::read_dir(model_output_root) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            if file_name
                .to_str()
                .is_some_and(|name| name.starts_with("oneshot_epoch_"))
            {
                return true;
            }
        }
    }

    false
}

pub fn all_expected_oneshot_model_outputs_exist(
    model_output_root: &str,
    num_oneshot_epochs: usize,
) -> bool {
    (1..=num_oneshot_epochs).all(|epoch_index| {
        Path::new(model_output_root)
            .join(format!("oneshot_epoch_{}/model", epoch_index))
            .exists()
    })
}

pub fn oneshot_epoch_model_ready(model_output_root: &str, epoch: usize) -> bool {
    let epoch_dir = Path::new(model_output_root).join(format!("oneshot_epoch_{epoch}"));
    epoch_dir.join("model").exists() && epoch_dir.join("training_summary.json").exists()
}

pub fn count_contiguous_ready_oneshot_epochs(
    model_output_root: &str,
    num_oneshot_epochs: usize,
) -> usize {
    let mut ready_count = 0usize;
    for epoch in 1..=num_oneshot_epochs {
        if oneshot_epoch_model_ready(model_output_root, epoch) {
            ready_count = epoch;
        } else {
            break;
        }
    }
    ready_count
}
