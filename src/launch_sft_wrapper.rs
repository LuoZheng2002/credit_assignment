use std::{path::Path, process::Stdio};

use research_utility::progress_tui_logger::log_info;
use tokio::process::Command;
use tokio::sync::watch;

use crate::{
    launch_backend_wrapper_shared::{
        bind_wrapper_tui_listener, socket_path_to_arg, spawn_wrapper_command,
        spawn_wrapper_tui_listener, write_json_payload_to_child_stdin,
    },
    python_training_config::PythonTrainingConfig,
};

pub async fn run_sft_wrapper_and_wait(
    num_gpus: usize,
    hf_model_name: &str,
    training_config: &PythonTrainingConfig,
    sft_training_data_path: &str,
    wrapper_log_path: &str,
) -> Result<(), String> {
    assert!(num_gpus > 0, "num_gpus must be positive");
    if !Path::new(sft_training_data_path).is_file() {
        return Err(format!(
            "SFT training data file does not exist: {}",
            sft_training_data_path
        ));
    }
    let (socket_path, listener) = bind_wrapper_tui_listener("sft")?;
    let socket_path_arg = socket_path_to_arg(&socket_path)?;

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("-m")
        .arg("src_py.wrappers.sft_wrapper")
        .arg("--num-gpus")
        .arg(num_gpus.to_string())
        .arg("--sft-training-data-path")
        .arg(sft_training_data_path)
        .arg("--hf-model-name")
        .arg(hf_model_name)
        .arg("--wrapper-log-path")
        .arg(wrapper_log_path)
        .arg("--orchestrator-socket-path")
        .arg(&socket_path_arg)
        .stdin(Stdio::piped());

    let mut process = spawn_wrapper_command(&mut command, &socket_path, "SFT wrapper")?;
    write_json_payload_to_child_stdin(&mut process, training_config, "SFT wrapper").await?;

    let (stop_signal_tx, stop_signal_rx) = watch::channel(false);
    let listener_handle =
        spawn_wrapper_tui_listener(listener, socket_path, "SFT wrapper", true, stop_signal_rx);

    let status = process
        .wait()
        .await
        .map_err(|err| format!("failed while waiting for SFT wrapper process: {}", err))?;

    let _ = stop_signal_tx.send(true);
    let _ = listener_handle.await;

    if status.success() {
        log_info(format!(
            "SFT wrapper completed successfully; details in {}",
            wrapper_log_path
        ));
        Ok(())
    } else {
        Err(format!(
            "SFT wrapper process exited with status {}; inspect log at {}",
            status, wrapper_log_path
        ))
    }
}
