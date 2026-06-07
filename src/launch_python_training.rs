use std::{path::Path, process::Stdio};

use research_utility::message::TuiMessage;
use research_utility::progress_tui_server::{log_info, log_message, log_warning};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;

pub struct PythonTrainingProcessHandle {
    pub process: Child,
    pub io_listener_task: JoinHandle<()>,
}

pub async fn launch_python_training_process(
    num_gpus: usize,
    training_config_path: String,
) -> PythonTrainingProcessHandle {
    assert!(num_gpus > 0, "num_gpus must be positive");
    let config_path = training_config_path.trim();
    assert!(
        !config_path.is_empty(),
        "training_config_path cannot be empty"
    );
    assert!(
        Path::new(config_path).is_file(),
        "training config file does not exist: {}",
        config_path
    );

    let master_port = std::env::var("MASTER_PORT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "29501".to_string());

    log_info(format!(
        "Launching python training with torchrun (num_gpus={}, master_port={})",
        num_gpus, master_port
    ));

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("torchrun")
        .arg("--nproc_per_node")
        .arg(num_gpus.to_string())
        .arg("--master_port")
        .arg(master_port)
        .arg("-m")
        .arg("src_py.train.main_from_config")
        .arg("--config-toml-path")
        .arg(config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut process = command.spawn().unwrap_or_else(|err| {
        panic!(
            "Failed to launch python training process with config {}: {}",
            config_path, err
        )
    });

    let stdout = process.stdout.take().unwrap_or_else(|| {
        panic!(
            "Failed to capture stdout for python training process ({})",
            config_path
        )
    });
    let stderr = process.stderr.take().unwrap_or_else(|| {
        panic!(
            "Failed to capture stderr for python training process ({})",
            config_path
        )
    });

    let stdout_listener = tokio::spawn(stream_process_output(stdout, false));
    let stderr_listener = tokio::spawn(stream_process_output(stderr, true));

    let io_listener_task = tokio::spawn(async move {
        let _ = stdout_listener.await;
        let _ = stderr_listener.await;
    });

    PythonTrainingProcessHandle {
        process,
        io_listener_task,
    }
}

async fn stream_process_output<R>(reader: R, is_stderr: bool)
where
    R: AsyncRead + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(content)) => {
                if content.is_empty() {
                    continue;
                }
                if is_stderr {
                    log_warning(format!("[TRAIN]: {}", content));
                } else if let Ok(parsed) = serde_json::from_str::<TuiMessage>(&content) {
                    log_message(parsed);
                } else {
                    log_info(format!("[TRAIN]: {}", content));
                }
            }
            Ok(None) => break,
            Err(err) => {
                log_warning(format!("[TRAIN]: Failed to read process output: {}", err));
                break;
            }
        }
    }
}
