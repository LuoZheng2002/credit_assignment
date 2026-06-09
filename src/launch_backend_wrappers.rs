use std::{
    path::Path,
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};

use research_utility::progress_tui_logger::{log_info, log_warning};
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::time::{Instant, sleep, timeout};

use crate::{compute_backend::ComputeBackend, launch_sglang_server::resolve_sglang_port};

#[derive(Debug, Clone, Deserialize)]
struct WrapperResultEvent {
    ok: bool,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    error_code: Option<String>,
    #[serde(default)]
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum WrapperEvent {
    #[serde(rename = "status")]
    Status {
        status: String,
        #[serde(default)]
        backend: Option<String>,
        #[serde(default)]
        message: Option<String>,
    },
    #[serde(rename = "result")]
    Result {
        ok: bool,
        #[serde(default)]
        backend: Option<String>,
        #[serde(default)]
        message: Option<String>,
        #[serde(default)]
        error_code: Option<String>,
        #[serde(default)]
        error_message: Option<String>,
    },
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        backend: Option<String>,
        #[serde(default)]
        error_code: Option<String>,
        #[serde(default)]
        error_message: Option<String>,
    },
}

pub async fn launch_inference_wrapper_process(
    compute_backend: ComputeBackend,
    model_path: Option<&str>,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    hf_model_name: &str,
    num_gpus: usize,
    log_path: Option<&str>,
) -> Result<(u16, Child), String> {
    let listen_port = resolve_sglang_port();
    let backend_flag = match compute_backend {
        ComputeBackend::Hpc => "hpc",
        ComputeBackend::Modal => "modal",
    };

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("-m")
        .arg("src_py.wrappers.inference_wrapper")
        .arg("--backend")
        .arg(backend_flag)
        .arg("--listen-port")
        .arg(listen_port.to_string())
        .arg("--num-gpus")
        .arg(num_gpus.to_string())
        .arg("--epoch")
        .arg(epoch.to_string())
        .arg("--model-cli-name")
        .arg(model_cli_name)
        .arg("--config-nickname")
        .arg(config_nickname)
        .arg("--hf-model-name")
        .arg(hf_model_name);

    if compute_backend == ComputeBackend::Hpc {
        let model_path = model_path.ok_or_else(|| {
            "HPC inference wrapper launch requires model_path to be provided".to_string()
        })?;
        command.arg("--model-path").arg(model_path);
    }

    if let Some(log_path) = log_path {
        let log_file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(log_path)
            .map_err(|err| format!("failed to open inference wrapper log path {}: {}", log_path, err))?;
        let log_file_err = log_file
            .try_clone()
            .map_err(|err| format!("failed to clone log file handle for {}: {}", log_path, err))?;
        command.stdout(Stdio::from(log_file));
        command.stderr(Stdio::from(log_file_err));
    } else {
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
    }

    let mut process = command
        .spawn()
        .map_err(|err| format!("failed to launch inference wrapper process: {}", err))?;

    if log_path.is_none() {
        if let Some(stdout) = process.stdout.take() {
            tokio::spawn(stream_output(stdout, false, None));
        }
        if let Some(stderr) = process.stderr.take() {
            tokio::spawn(stream_output(stderr, true, None));
        }
    }

    wait_for_wrapper_health(listen_port, &mut process).await?;
    Ok((listen_port, process))
}

pub async fn run_training_wrapper_and_wait(
    compute_backend: ComputeBackend,
    num_gpus: usize,
    hf_model_name: &str,
    training_config_json: String,
    trajectory_sqlite_path: &str,
) -> Result<(), String> {
    assert!(num_gpus > 0, "num_gpus must be positive");
    if compute_backend == ComputeBackend::Modal && num_gpus != 1 {
        return Err(format!(
            "Modal backend requires num_gpus=1 (one H100 per experiment), got {}",
            num_gpus
        ));
    }
    if !Path::new(trajectory_sqlite_path).is_file() {
        return Err(format!(
            "training trajectory sqlite path does not exist: {}",
            trajectory_sqlite_path
        ));
    }
    let backend_flag = match compute_backend {
        ComputeBackend::Hpc => "hpc",
        ComputeBackend::Modal => "modal",
    };

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("-m")
        .arg("src_py.wrappers.training_wrapper")
        .arg("--backend")
        .arg(backend_flag)
        .arg("--num-gpus")
        .arg(num_gpus.to_string())
        .arg("--training-config-json")
        .arg(training_config_json)
        .arg("--trajectory-sqlite-path")
        .arg(trajectory_sqlite_path)
        .arg("--hf-model-name")
        .arg(hf_model_name)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut process = command
        .spawn()
        .map_err(|err| format!("failed to launch training wrapper process: {}", err))?;

    let stdout = process
        .stdout
        .take()
        .ok_or_else(|| "failed to capture training wrapper stdout".to_string())?;
    let stderr = process
        .stderr
        .take()
        .ok_or_else(|| "failed to capture training wrapper stderr".to_string())?;

    let wrapper_result = Arc::new(Mutex::new(None::<WrapperResultEvent>));

    let stdout_listener = tokio::spawn(stream_output(
        stdout,
        false,
        Some(wrapper_result.clone()),
    ));
    let stderr_listener = tokio::spawn(stream_output(stderr, true, None));

    let status = process
        .wait()
        .await
        .map_err(|err| format!("failed while waiting for training wrapper process: {}", err))?;

    let _ = stdout_listener.await;
    let _ = stderr_listener.await;

    let result_event = wrapper_result
        .lock()
        .map_err(|_| "failed to lock wrapper result mutex".to_string())?
        .clone();

    match (status.success(), result_event) {
        (true, Some(result)) if result.ok => {
            if let Some(message) = result.message {
                log_info(format!("Training wrapper reported success: {}", message));
            }
            Ok(())
        }
        (_, Some(result)) => Err(format!(
            "training wrapper failed (backend={}, code={}, message={}, process_status={})",
            result.backend.unwrap_or_else(|| "unknown".to_string()),
            result.error_code.unwrap_or_else(|| "unknown".to_string()),
            result
                .error_message
                .or(result.message)
                .unwrap_or_else(|| "unknown".to_string()),
            status
        )),
        (true, None) => Err("training wrapper exited successfully but emitted no result event".to_string()),
        (false, None) => Err(format!(
            "training wrapper process exited with status {} and emitted no result event",
            status
        )),
    }
}

async fn stream_output<R>(
    reader: R,
    is_stderr: bool,
    result_sink: Option<Arc<Mutex<Option<WrapperResultEvent>>>>,
)
where
    R: AsyncRead + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(content)) = lines.next_line().await {
        if content.is_empty() {
            continue;
        }
        if is_stderr {
            log_warning(format!("[WRAPPER]: {}", content));
        } else {
            if let Ok(event) = serde_json::from_str::<WrapperEvent>(&content) {
                match event {
                    WrapperEvent::Status {
                        status,
                        backend,
                        message,
                    } => {
                        log_info(format!(
                            "[WRAPPER][status] backend={} status={} message={}",
                            backend.unwrap_or_else(|| "unknown".to_string()),
                            status,
                            message.unwrap_or_default()
                        ));
                    }
                    WrapperEvent::Result {
                        ok,
                        backend,
                        message,
                        error_code,
                        error_message,
                    } => {
                        if let Some(sink) = &result_sink {
                            if let Ok(mut guard) = sink.lock() {
                                *guard = Some(WrapperResultEvent {
                                    ok,
                                    backend: backend.clone(),
                                    message: message.clone(),
                                    error_code: error_code.clone(),
                                    error_message: error_message.clone(),
                                });
                            }
                        }
                        log_info(format!(
                            "[WRAPPER][result] backend={} ok={} code={} message={}",
                            backend.unwrap_or_else(|| "unknown".to_string()),
                            ok,
                            error_code.unwrap_or_else(|| "none".to_string()),
                            error_message.or(message).unwrap_or_default()
                        ));
                    }
                    WrapperEvent::Error {
                        backend,
                        error_code,
                        error_message,
                    } => {
                        log_warning(format!(
                            "[WRAPPER][error] backend={} code={} message={}",
                            backend.unwrap_or_else(|| "unknown".to_string()),
                            error_code.unwrap_or_else(|| "unknown".to_string()),
                            error_message.unwrap_or_default()
                        ));
                    }
                }
                continue;
            }
            log_info(format!("[WRAPPER]: {}", content));
        }
    }
}

async fn wait_for_wrapper_health(port: u16, process: &mut Child) -> Result<(), String> {
    let timeout_duration = Duration::from_secs(180);
    let sleep_interval = Duration::from_millis(500);
    let start = Instant::now();
    let url = format!("http://127.0.0.1:{}/health", port);

    loop {
        if let Ok(Some(status)) = process.try_wait() {
            return Err(format!(
                "inference wrapper process exited before becoming healthy: {}",
                status
            ));
        }

        match timeout(Duration::from_secs(2), reqwest::get(url.clone())).await {
            Ok(Ok(response)) if response.status().is_success() => return Ok(()),
            _ => {}
        }

        if start.elapsed() >= timeout_duration {
            return Err(format!(
                "timed out waiting for inference wrapper health endpoint ({})",
                url
            ));
        }
        sleep(sleep_interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::WrapperEvent;

    #[test]
    fn wrapper_event_status_parses() {
        let json = r#"{"type":"status","backend":"modal","status":"running","message":"training"}"#;
        let parsed = serde_json::from_str::<WrapperEvent>(json).expect("status event should parse");
        match parsed {
            WrapperEvent::Status {
                status,
                backend,
                message,
            } => {
                assert_eq!(status, "running");
                assert_eq!(backend.as_deref(), Some("modal"));
                assert_eq!(message.as_deref(), Some("training"));
            }
            _ => panic!("expected status event"),
        }
    }

    #[test]
    fn wrapper_event_result_parses() {
        let json = r#"{"type":"result","backend":"hpc","ok":true,"message":"done"}"#;
        let parsed = serde_json::from_str::<WrapperEvent>(json).expect("result event should parse");
        match parsed {
            WrapperEvent::Result {
                ok,
                backend,
                message,
                error_code,
                error_message,
            } => {
                assert!(ok);
                assert_eq!(backend.as_deref(), Some("hpc"));
                assert_eq!(message.as_deref(), Some("done"));
                assert!(error_code.is_none());
                assert!(error_message.is_none());
            }
            _ => panic!("expected result event"),
        }
    }

    #[test]
    fn wrapper_event_error_parses() {
        let json = r#"{"type":"error","backend":"hpc","error_code":"TRAIN_PROCESS_FAILED","error_message":"boom"}"#;
        let parsed = serde_json::from_str::<WrapperEvent>(json).expect("error event should parse");
        match parsed {
            WrapperEvent::Error {
                backend,
                error_code,
                error_message,
            } => {
                assert_eq!(backend.as_deref(), Some("hpc"));
                assert_eq!(error_code.as_deref(), Some("TRAIN_PROCESS_FAILED"));
                assert_eq!(error_message.as_deref(), Some("boom"));
            }
            _ => panic!("expected error event"),
        }
    }
}
