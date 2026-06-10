use std::{path::Path, process::Stdio};

use research_utility::message::TuiMessage;
use research_utility::progress_tui_logger::{log_info, log_message, log_warning};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;

pub struct PythonTrainingProcessHandle {
    pub process: Child,
    pub io_listener_task: JoinHandle<()>,
}

pub async fn launch_python_training_process(
    num_gpus: usize,
    training_job_folder_path: String,
) -> PythonTrainingProcessHandle {
    assert!(num_gpus > 0, "num_gpus must be positive");
    let job_folder_path = training_job_folder_path.trim();
    assert!(
        !job_folder_path.is_empty(),
        "training_job_folder_path cannot be empty"
    );
    assert!(
        Path::new(job_folder_path).is_dir(),
        "training job folder does not exist: {}",
        job_folder_path
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
        .arg("src_py.train.main")
        .arg("--job-folder-path")
        .arg(job_folder_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut process = command.spawn().unwrap_or_else(|err| {
        panic!(
            "Failed to launch python training process with job folder {}: {}",
            job_folder_path, err
        )
    });

    let stdout = process.stdout.take().unwrap_or_else(|| {
        panic!(
            "Failed to capture stdout for python training process ({})",
            job_folder_path
        )
    });
    let stderr = process.stderr.take().unwrap_or_else(|| {
        panic!(
            "Failed to capture stderr for python training process ({})",
            job_folder_path
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
