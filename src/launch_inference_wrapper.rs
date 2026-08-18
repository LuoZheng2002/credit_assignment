use std::{fs, net::SocketAddr, path::Path, time::Duration};

use clap::ValueEnum;
use research_utility::progress_text_logger::{log_info, log_warning};
use tokio::process::{Child, Command};
use tokio::time::{Instant, sleep, timeout};

use serde::Serialize;

use research_utility::launch_python_process::{PythonProcessHandle, PythonProcessLauncher};

#[derive(Clone, Copy, Debug, Serialize, serde::Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum InferenceBackend {
    Sglang,
    Vllm,
}

impl std::fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sglang => write!(f, "sglang"),
            Self::Vllm => write!(f, "vllm"),
        }
    }
}

#[derive(Serialize)]
struct InferenceWrapperArgs {
    listen_port: u16,
    inference_backend: InferenceBackend,
    num_gpus: usize,
    epoch: usize,
    model_cli_name: String,
    config_nickname: String,
    hf_model_name: String,
    model_path: String,
    wrapper_log_path: String,
}

pub async fn launch_inference_wrapper_process(
    inference_backend: InferenceBackend,
    model_path: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    hf_model_name: &str,
    num_gpus: usize,
    wrapper_log_path: &str,
) -> Result<(u16, PythonProcessHandle), String> {
    let listen_port = configured_wrapper_port()?;
    ensure_wrapper_port_available(listen_port).await?;

    if let Some(parent) = Path::new(wrapper_log_path).parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "Failed to create parent directory for inference wrapper log path {}: {}",
                wrapper_log_path, err
            )
        });
    }

    let args = InferenceWrapperArgs {
        listen_port,
        inference_backend,
        num_gpus,
        epoch,
        model_cli_name: model_cli_name.to_string(),
        config_nickname: config_nickname.to_string(),
        hf_model_name: hf_model_name.to_string(),
        model_path: model_path.to_string(),
        wrapper_log_path: wrapper_log_path.to_string(),
    };

    let mut handle = PythonProcessLauncher::new("inference", "src_py.wrappers.inference_wrapper")
        .with_stdin_json(&args)?
        .with_process_group(true)
        .launch()
        .await?;

    wait_for_wrapper_health(listen_port, Some(&mut handle.child), wrapper_log_path).await?;
    Ok((listen_port, handle))
}

pub async fn update_inference_model(
    port: u16,
    model_path: &str,
    wrapper_log_path: &str,
) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{}/update_model", port);
    log_info(format!(
        "Updating inference model to {} via {}",
        model_path, url
    ));
    let payload = serde_json::json!({"model_path": model_path});

    let client = reqwest::Client::new();
    let timeout_secs = std::env::var("INFERENCE_UPDATE_MODEL_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1800);
    let response = timeout(
        Duration::from_secs(timeout_secs),
        client.post(&url).json(&payload).send(),
    )
    .await
    .map_err(|_| {
        format!(
            "update_model request to {} timed out after {}s",
            url, timeout_secs
        )
    })?
    .map_err(|e| format!("update_model request to {} failed: {}", url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("update_model returned {}: {}", status, body));
    }

    log_info(format!(
        "Model update succeeded; re-verifying health at port {}",
        port
    ));
    wait_for_wrapper_health(port, None, wrapper_log_path).await?;
    log_info("Model update completed and health check passed");
    Ok(())
}

pub async fn shut_down_inference_wrapper_process(process: &mut Child) {
    match process.try_wait() {
        Ok(Some(status)) => {
            log_info(format!(
                "Inference wrapper shutdown: process already exited before shutdown call (status={})",
                status
            ));
        }
        Ok(None) => {
            let pid_for_log = process.id();
            log_info(format!(
                "Inference wrapper shutdown: process is running before shutdown (pid={:?}), sending TERM",
                pid_for_log
            ));
            if let Some(pid) = process.id() {
                send_signal_to_process_group(pid, "-TERM").await;
            }

            match timeout(Duration::from_secs(15), process.wait()).await {
                Ok(Ok(status)) => {
                    log_info(format!(
                        "Inference wrapper shutdown: exited after TERM (pid={:?}, status={})",
                        pid_for_log, status
                    ));
                    return;
                }
                Ok(Err(err)) => panic!("Failed to wait on inference wrapper process: {}", err),
                Err(_) => {
                    log_info(format!(
                        "Inference wrapper shutdown: TERM timed out after 15s (pid={:?}), escalating to KILL",
                        pid_for_log
                    ));
                }
            }

            if let Some(pid) = process.id() {
                send_signal_to_process_group(pid, "-KILL").await;
                match timeout(Duration::from_secs(10), process.wait()).await {
                    Ok(Ok(status)) => {
                        log_info(format!(
                            "Inference wrapper shutdown: exited after KILL (pid={:?}, status={})",
                            pid_for_log, status
                        ));
                        return;
                    }
                    Ok(Err(err)) => {
                        panic!("Failed to wait on inference wrapper process: {}", err)
                    }
                    Err(_) => {
                        log_info(format!(
                            "Inference wrapper shutdown: KILL wait timed out after 10s (pid={:?}), falling back to process.kill()",
                            pid_for_log
                        ));
                    }
                }
            }

            log_info(format!(
                "Inference wrapper shutdown: invoking process.kill() fallback (pid={:?})",
                pid_for_log
            ));
            if let Err(err) = process.kill().await {
                panic!("Failed to kill inference wrapper process: {}", err);
            }
        }
        Err(err) => panic!(
            "Failed to inspect inference wrapper process status: {}",
            err
        ),
    }

    if let Err(err) = process.wait().await {
        panic!("Failed to wait on inference wrapper process: {}", err);
    }
}

pub async fn best_effort_shutdown_stale_inference_wrapper() {
    let port = match configured_wrapper_port() {
        Ok(port) => port,
        Err(err) => {
            log_warning(format!(
                "Inference wrapper stale-shutdown check skipped: {}",
                err
            ));
            return;
        }
    };
    if !is_port_listening(port).await {
        log_info(format!(
            "Inference wrapper stale-shutdown check: no listener on configured port {}, nothing to clean up",
            port
        ));
        return;
    }

    log_info(format!(
        "Inference wrapper stale-shutdown check: detected listener on configured port {}, attempting pkill cleanup",
        port
    ));
    let pattern = format!("src_py.wrappers.inference_wrapper.*--listen-port {}", port);
    match Command::new("pkill").arg("-f").arg(&pattern).status().await {
        Ok(status) => {
            log_info(format!(
                "Inference wrapper stale-shutdown check: pkill finished with status {} for pattern '{}'",
                status, pattern
            ));
        }
        Err(err) => {
            log_info(format!(
                "Inference wrapper stale-shutdown check: failed to execute pkill for pattern '{}': {}",
                pattern, err
            ));
            return;
        }
    }

    for _ in 0..20 {
        if !is_port_listening(port).await {
            log_info(format!(
                "Inference wrapper stale-shutdown check: port {} is no longer listening after cleanup",
                port
            ));
            return;
        }
        sleep(Duration::from_millis(250)).await;
    }

    log_info(format!(
        "Inference wrapper stale-shutdown check: port {} is still listening after pkill cleanup",
        port
    ));
}

async fn wait_for_wrapper_health(
    port: u16,
    mut handle: Option<&mut Child>,
    log_path: &str,
) -> Result<(), String> {
    let timeout_secs = std::env::var("INFERENCE_WRAPPER_HEALTH_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1800);
    let timeout_duration = Duration::from_secs(timeout_secs);
    let sleep_interval = Duration::from_millis(500);
    let start = Instant::now();
    let url = format!("http://127.0.0.1:{}/health", port);

    loop {
        if let Some(proc) = handle.as_mut() {
            if let Ok(Some(status)) = proc.try_wait() {
                return Err(format!(
                    "inference wrapper process exited before becoming healthy: {}; inspect wrapper log at {}",
                    status, log_path
                ));
            }
        }

        match timeout(Duration::from_secs(2), reqwest::get(url.clone())).await {
            Ok(Ok(response)) if response.status().is_success() => {
                if let Ok(body) = response.json::<serde_json::Value>().await {
                    if body
                        .get("status")
                        .and_then(serde_json::Value::as_str)
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
        "wrapper port {} is already in use before launch (likely stale process). Please free the port or set INFERENCE_WRAPPER_PORT to an unused value",
        port
    ))
}

fn configured_wrapper_port() -> Result<u16, String> {
    match std::env::var("INFERENCE_WRAPPER_PORT") {
        Ok(value) => {
            let port = value.parse::<u16>().map_err(|err| {
                format!(
                    "INFERENCE_WRAPPER_PORT must be a valid u16 port, got {:?}: {}",
                    value, err
                )
            })?;
            if port == 0 {
                return Err("INFERENCE_WRAPPER_PORT must not be 0".to_string());
            }
            Ok(port)
        }
        Err(std::env::VarError::NotPresent) => Ok(30000u16),
        Err(err) => Err(format!("failed to read INFERENCE_WRAPPER_PORT: {}", err)),
    }
}

async fn send_signal_to_process_group(pid: u32, signal: &str) {
    let status = Command::new("kill")
        .arg(signal)
        .arg("--")
        .arg(format!("-{}", pid))
        .status()
        .await;
    match status {
        Ok(_) => {}
        Err(err) => {
            panic!(
                "Failed to send signal {} to process group {}: {}",
                signal, pid, err
            );
        }
    }
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
