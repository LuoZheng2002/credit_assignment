use std::{fs, net::SocketAddr, path::Path, time::Duration};

use research_utility::progress_tui_logger::{log_info, log_warning};
use tokio::process::{Child, Command};
use tokio::sync::watch;
use tokio::time::{Instant, sleep, timeout};

use crate::{
    launch_backend_wrapper_shared::{
        bind_wrapper_tui_listener, socket_path_to_arg, spawn_wrapper_command,
        spawn_wrapper_tui_listener,
    },
    llm_model::LlmModelMarker,
};

pub fn model_uses_sglang<M: LlmModelMarker>() -> bool {
    !M::CLI_NAME.starts_with("gpt-")
}

pub async fn launch_inference_wrapper_process(
    model_path: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    hf_model_name: &str,
    num_gpus: usize,
    wrapper_log_path: &str,
) -> Result<(u16, Child, watch::Sender<bool>, tokio::task::JoinHandle<()>), String> {
    let listen_port = 30000u16;
    ensure_wrapper_port_available(listen_port).await?;
    let (socket_path, listener) = bind_wrapper_tui_listener("inference")?;
    let socket_path_arg = socket_path_to_arg(&socket_path)?;

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("-m")
        .arg("src_py.wrappers.inference_wrapper")
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
        .arg("--orchestrator-socket-path")
        .arg(&socket_path_arg)
        .arg("--hf-model-name")
        .arg(hf_model_name)
        .arg("--model-path")
        .arg(model_path);

    command.arg("--wrapper-log-path").arg(wrapper_log_path);
    if let Some(parent) = Path::new(wrapper_log_path).parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "Failed to create parent directory for inference wrapper log path {}: {}",
                wrapper_log_path, err
            )
        });
    }

    #[cfg(unix)]
    command.process_group(0);

    let mut process = spawn_wrapper_command(&mut command, &socket_path, "inference wrapper")?;

    let (stop_signal_tx, stop_signal_rx) = watch::channel(false);
    let listener_handle = spawn_wrapper_tui_listener(
        listener,
        socket_path,
        "inference wrapper",
        false,
        stop_signal_rx,
    );

    wait_for_wrapper_health(listen_port, &mut process, wrapper_log_path).await?;
    Ok((listen_port, process, stop_signal_tx, listener_handle))
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
    let port = 30000u16;
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
    process: &mut Child,
    log_path: &str,
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
            return Err(format!(
                "inference wrapper process exited before becoming healthy: {}; inspect wrapper log at {}",
                status, log_path
            ));
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
        "wrapper port {} is already in use before launch (likely stale process). Please free the port or set SGLANG_PORT to an unused value",
        port
    ))
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
