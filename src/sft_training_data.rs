use std::fs::{self, OpenOptions};
use std::io::Write;

use research_utility::progress_tui_logger::log_info;
use serde::{Deserialize, Serialize};

use crate::hybrid_sft_dataset::open_hybrid_sft_dataset;
use crate::utils::mount_dir;

/// A raw-text SFT entry stored in msgpack for Python-side tokenization.
#[derive(Serialize, Deserialize)]
struct SftRawEntry {
    prompt: String,
    response: String,
}

pub fn sft_training_data_file_path(model_cli_name: &str, config_nickname: &str) -> String {
    let mount = mount_dir().expect("mount_dir must be configured");
    format!(
        "{}/results/{}/{}/sft_training_data.msgpack",
        mount, model_cli_name, config_nickname
    )
}

pub fn generate_sft_training_data(
    model_cli_name: &str,
    config_nickname: &str,
) -> Result<(), String> {
    let store = open_hybrid_sft_dataset();
    let n = store.len();
    assert!(n > 0, "SFT dataset must be non-empty");

    log_info(format!(
        "Generating SFT raw-text data from {} dataset entries (tokenization deferred to Python)",
        n
    ));

    let file_path = sft_training_data_file_path(model_cli_name, config_nickname);
    if let Some(parent) = std::path::Path::new(&file_path).parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create parent directory for SFT training data at {}: {}",
                parent.display(),
                err
            )
        })?;
    }
    // Remove existing file so we start fresh
    let _ = fs::remove_file(&file_path);

    let path = std::path::Path::new(&file_path);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| {
            format!(
                "Failed to create SFT training data msgpack file {}: {}",
                path.display(),
                err
            )
        })?;

    for result in store.iter()? {
        let (_idx, entry) = result?;
        let raw_entry = SftRawEntry {
            prompt: entry.prompt.clone(),
            response: entry.reference_trajectory.clone(),
        };
        let bytes = rmp_serde::to_vec_named(&raw_entry).map_err(|err| {
            format!(
                "Failed to serialize SFT raw entry for msgpack file {}: {}",
                path.display(),
                err
            )
        })?;
        file.write_all(&bytes).map_err(|err| {
            format!(
                "Failed to write SFT raw entry to msgpack file {}: {}",
                path.display(),
                err
            )
        })?;
    }

    log_info(format!(
        "Generated {} SFT raw-text samples at {}",
        n, file_path
    ));

    Ok(())
}
