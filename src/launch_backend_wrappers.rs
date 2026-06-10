use std::{
    net::SocketAddr,
    path::Path,
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};

use research_utility::progress_tui_logger::{log_info, log_warning};
use serde::Deserialize;
use serde_json::Value;
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
        #[serde(default)]
        metrics: Option<Value>,
        #[serde(default)]
        model_official_name: Option<String>,
        #[serde(default)]
        config_nickname: Option<String>,
        #[serde(default)]
        epoch: Option<usize>,
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
    artifact_root_dir: Option<&str>,
    modal_base_url: Option<&str>,
    modal_auth_token_env_var: Option<&str>,
    num_gpus: usize,
    log_path: Option<&str>,
) -> Result<(u16, Child), String> {
    let listen_port = resolve_sglang_port();
    ensure_wrapper_port_available(listen_port).await?;
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

    if let Some(artifact_root_dir) = artifact_root_dir {
        command
            .arg("--artifact-root-dir")
            .arg(artifact_root_dir);
    }

    if let Some(modal_base_url) = modal_base_url {
        command.arg("--modal-base-url").arg(modal_base_url);
    }

    if let Some(modal_auth_token_env_var) = modal_auth_token_env_var {
        command
            .arg("--modal-auth-token-env-var")
            .arg(modal_auth_token_env_var);
    }

    if let Some(log_path) = log_path {
        command.arg("--wrapper-log-path").arg(log_path);
    }

    let model_path = model_path.ok_or_else(|| {
        "Inference wrapper launch requires model_path to be provided".to_string()
    })?;
    command.arg("--model-path").arg(model_path);

    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut process = command
        .spawn()
        .map_err(|err| format!("failed to launch inference wrapper process: {}", err))?;

    if let Some(stdout) = process.stdout.take() {
        tokio::spawn(stream_output(stdout, false, None));
    }
    if let Some(stderr) = process.stderr.take() {
        tokio::spawn(stream_output(stderr, true, None));
    }

    wait_for_wrapper_health(listen_port, &mut process, log_path).await?;
    Ok((listen_port, process))
}

pub async fn run_training_wrapper_and_wait(
    compute_backend: ComputeBackend,
    num_gpus: usize,
    hf_model_name: &str,
    training_config_json: String,
    trajectory_sqlite_path: &str,
    training_wrapper_log_path: Option<&str>,
) -> Result<(), String> {
    assert!(num_gpus > 0, "num_gpus must be positive");
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
    let active_log_path = training_wrapper_log_path.and_then(|path| {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    });

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
        .arg("--training-wrapper-log-path")
        .arg(training_wrapper_log_path.unwrap_or(""));

    if let Some(log_path) = active_log_path {
        let log_file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(log_path)
            .map_err(|err| format!("failed to open training wrapper log path {}: {}", log_path, err))?;
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
        .map_err(|err| format!("failed to launch training wrapper process: {}", err))?;

    let wrapper_result = Arc::new(Mutex::new(None::<WrapperResultEvent>));
    let mut stdout_listener = None;
    let mut stderr_listener = None;
    if active_log_path.is_none() {
        let stdout = process
            .stdout
            .take()
            .ok_or_else(|| "failed to capture training wrapper stdout".to_string())?;
        let stderr = process
            .stderr
            .take()
            .ok_or_else(|| "failed to capture training wrapper stderr".to_string())?;
        stdout_listener = Some(tokio::spawn(stream_output(
            stdout,
            false,
            Some(wrapper_result.clone()),
        )));
        stderr_listener = Some(tokio::spawn(stream_output(stderr, true, None)));
    }

    let status = process
        .wait()
        .await
        .map_err(|err| format!("failed while waiting for training wrapper process: {}", err))?;

    if let Some(listener) = stdout_listener {
        let _ = listener.await;
    }
    if let Some(listener) = stderr_listener {
        let _ = listener.await;
    }

    if let Some(log_path) = active_log_path {
        if status.success() {
            log_info(format!(
                "Training wrapper completed successfully; details in {}",
                log_path
            ));
            return Ok(());
        }
        return Err(format!(
            "training wrapper process exited with status {}; inspect log at {}",
            status, log_path
        ));
    }

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
                        metrics,
                        model_official_name,
                        config_nickname,
                        epoch,
                    } => {
                        let metrics_text = metrics
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "none".to_string());
                        let identity_text = format!(
                            "model_official_name={} config_nickname={} epoch={}",
                            model_official_name.unwrap_or_else(|| "none".to_string()),
                            config_nickname.unwrap_or_else(|| "none".to_string()),
                            epoch
                                .map(|value| value.to_string())
                                .unwrap_or_else(|| "none".to_string()),
                        );
                        log_info(format!(
                            "[WRAPPER][status] backend={} status={} message={} metrics={} {}",
                            backend.unwrap_or_else(|| "unknown".to_string()),
                            status,
                            message.unwrap_or_default(),
                            metrics_text,
                            identity_text,
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

async fn wait_for_wrapper_health(
    port: u16,
    process: &mut Child,
    log_path: Option<&str>,
) -> Result<(), String> {
    let timeout_secs = std::env::var("INFERENCE_WRAPPER_HEALTH_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(900);
    let timeout_duration = Duration::from_secs(timeout_secs);
    let sleep_interval = Duration::from_millis(500);
    let start = Instant::now();
    let url = format!("http://127.0.0.1:{}/health", port);

    loop {
        if let Ok(Some(status)) = process.try_wait() {
            let hint = log_path
                .map(|path| format!("; inspect wrapper log at {}", path))
                .unwrap_or_default();
            return Err(format!(
                "inference wrapper process exited before becoming healthy: {}{}",
                status, hint
            ));
        }

        match timeout(Duration::from_secs(2), reqwest::get(url.clone())).await {
            Ok(Ok(response)) if response.status().is_success() => {
                if let Ok(body) = response.json::<Value>().await {
                    if body
                        .get("status")
                        .and_then(Value::as_str)
                        .map(|s| s == "ok")
                        .unwrap_or(false)
                    {
                        return Ok(());
                    }
                }
            }
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

async fn ensure_wrapper_port_available(port: u16) -> Result<(), String> {
    if !is_port_listening(port).await {
        return Ok(());
    }
    log_warning(format!(
        "Detected existing listener on wrapper port {}; attempting stale wrapper cleanup",
        port
    ));
    let pattern = format!("src_py.wrappers.inference_wrapper.*--listen-port {}", port);
    match Command::new("pkill").arg("-f").arg(&pattern).status().await {
        Ok(status) => {
            log_info(format!(
                "Stale wrapper cleanup command completed (pattern='{}', status={})",
                pattern, status
            ));
        }
        Err(err) => {
            return Err(format!(
                "wrapper port {} is already in use and stale-wrapper cleanup failed to execute (pattern='{}'): {}",
                port, pattern, err
            ));
        }
    }
    for _ in 0..20 {
        if !is_port_listening(port).await {
            log_info(format!(
                "Stale wrapper cleanup succeeded; wrapper port {} is now free",
                port
            ));
            return Ok(());
        }
        sleep(Duration::from_millis(250)).await;
    }
    Err(format!(
        "wrapper port {} is already in use before launch (likely stale process). Please free the port or set SGLANG_PORT to an unused value",
        port
    ))
}

async fn is_port_listening(port: u16) -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    timeout(
        Duration::from_millis(250),
        tokio::net::TcpStream::connect(address),
    )
    .await
    .is_ok_and(|result| result.is_ok())
}

#[cfg(test)]
mod tests {
    use super::WrapperEvent;

    #[test]
    fn wrapper_event_status_parses() {
        let json = r#"{"type":"status","backend":"modal","status":"running","message":"training","metrics":{"containers_running":1},"model_official_name":"Qwen/Qwen2.5-7B-Instruct","config_nickname":"notool","epoch":0}"#;
        let parsed = serde_json::from_str::<WrapperEvent>(json).expect("status event should parse");
        match parsed {
            WrapperEvent::Status {
                status,
                backend,
                message,
                metrics,
                model_official_name,
                config_nickname,
                epoch,
            } => {
                assert_eq!(status, "running");
                assert_eq!(backend.as_deref(), Some("modal"));
                assert_eq!(message.as_deref(), Some("training"));
                assert!(metrics.is_some());
                assert_eq!(model_official_name.as_deref(), Some("Qwen/Qwen2.5-7B-Instruct"));
                assert_eq!(config_nickname.as_deref(), Some("notool"));
                assert_eq!(epoch, Some(0));
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
