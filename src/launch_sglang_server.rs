use std::{
    fs::{self, OpenOptions},
    net::SocketAddr,
    path::Path,
    process::Stdio,
    time::Duration,
};

use research_utility::progress_tui_server::log_info;
use tokio::process::{Child, Command};
use tokio::time::{Instant, sleep, timeout};

use crate::{constants::sglang_context_length, llm_model::LlmModelMarker};

pub fn model_uses_sglang<M: LlmModelMarker>() -> bool {
    !M::CLI_NAME.starts_with("gpt-")
}

pub async fn launch_sglang_server_process<M: LlmModelMarker>(
    model_path: &str,
    num_gpus: usize,
    use_tool: bool,
    sglang_server_log_path: Option<&str>,
) -> Result<(u16, Child), String> {
    assert!(num_gpus > 0, "num_gpus must be positive");
    let sglang_port = resolve_sglang_port();
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("--project")
        .arg("pyprojects/sglang")
        .arg("python")
        .arg("-m")
        .arg("sglang.launch_server")
        .arg("--model-path")
        .arg(model_path)
        .arg("--host")
        .arg("0.0.0.0")
        .arg("--port")
        .arg(sglang_port.to_string())
        .arg("--dp")
        .arg(num_gpus.to_string())
        .arg("--context-length")
        .arg(sglang_context_length(use_tool).to_string())
        // .arg("--load-balance-method")
        // .arg("total_tokens")
        // .arg("--enable-mixed-chunk")
        // .arg("--schedule-policy")
        // .arg("lpm")
        ;

    #[cfg(unix)]
    command.process_group(0);

    if let Some(log_path) = sglang_server_log_path {
        if let Some(parent) = Path::new(log_path).parent() {
            fs::create_dir_all(parent).unwrap_or_else(|err| {
                panic!(
                    "Failed to create parent directory for sglang server log path {}: {}",
                    log_path, err
                )
            });
        }
        let stdout_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(log_path)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to open sglang server log path {}: {}",
                    log_path, err
                )
            });
        let stderr_file = stdout_file.try_clone().unwrap_or_else(|err| {
            panic!(
                "Failed to clone file handle for sglang server log path {}: {}",
                log_path, err
            )
        });
        command.stdout(Stdio::from(stdout_file));
        command.stderr(Stdio::from(stderr_file));
    }

    let mut process = command.spawn().unwrap_or_else(|err| {
        panic!(
            "Failed to launch sglang inference server for model {}: {}",
            M::CLI_NAME,
            err
        )
    });

    wait_for_sglang_ready(sglang_port, &mut process).await?;
    Ok((sglang_port, process))
}

pub async fn shut_down_sglang_server_process(process: &mut Child) {
    match process.try_wait() {
        Ok(Some(status)) => {
            log_info(format!(
                "SGLang shutdown: process already exited before shutdown call (status={})",
                status
            ));
        }
        Ok(None) => {
            let pid_for_log = process.id();
            log_info(format!(
                "SGLang shutdown: process is running before shutdown (pid={:?}), sending TERM",
                pid_for_log
            ));
            if let Some(pid) = process.id() {
                send_signal_to_process_group(pid, "-TERM").await;
            }

            match timeout(Duration::from_secs(15), process.wait()).await {
                Ok(Ok(status)) => {
                    log_info(format!(
                        "SGLang shutdown: exited after TERM (pid={:?}, status={})",
                        pid_for_log, status
                    ));
                    return;
                }
                Ok(Err(err)) => panic!("Failed to wait on inference server process: {}", err),
                Err(_) => {
                    log_info(format!(
                        "SGLang shutdown: TERM timed out after 15s (pid={:?}), escalating to KILL",
                        pid_for_log
                    ));
                }
            }

            if let Some(pid) = process.id() {
                send_signal_to_process_group(pid, "-KILL").await;
                match timeout(Duration::from_secs(10), process.wait()).await {
                    Ok(Ok(status)) => {
                        log_info(format!(
                            "SGLang shutdown: exited after KILL (pid={:?}, status={})",
                            pid_for_log, status
                        ));
                        return;
                    }
                    Ok(Err(err)) => {
                        panic!("Failed to wait on inference server process: {}", err)
                    }
                    Err(_) => {
                        log_info(format!(
                            "SGLang shutdown: KILL wait timed out after 10s (pid={:?}), falling back to process.kill()",
                            pid_for_log
                        ));
                    }
                }
            }

            log_info(format!(
                "SGLang shutdown: invoking process.kill() fallback (pid={:?})",
                pid_for_log
            ));
            if let Err(err) = process.kill().await {
                panic!("Failed to kill inference server process: {}", err);
            }
        }
        Err(err) => panic!("Failed to inspect inference server process status: {}", err),
    }

    if let Err(err) = process.wait().await {
        panic!("Failed to wait on inference server process: {}", err);
    }
}

pub async fn best_effort_shutdown_stale_sglang_server() {
    let port = resolve_sglang_port();
    if !is_port_listening(port).await {
        log_info(format!(
            "SGLang stale-shutdown check: no listener on configured port {}, nothing to clean up",
            port
        ));
        return;
    }

    log_info(format!(
        "SGLang stale-shutdown check: detected listener on configured port {}, attempting pkill cleanup",
        port
    ));
    let pattern = format!("sglang.launch_server.*--port {}", port);
    match Command::new("pkill").arg("-f").arg(&pattern).status().await {
        Ok(status) => {
            log_info(format!(
                "SGLang stale-shutdown check: pkill finished with status {} for pattern '{}'",
                status, pattern
            ));
        }
        Err(err) => {
            log_info(format!(
                "SGLang stale-shutdown check: failed to execute pkill for pattern '{}': {}",
                pattern, err
            ));
            return;
        }
    }

    for _ in 0..20 {
        if !is_port_listening(port).await {
            log_info(format!(
                "SGLang stale-shutdown check: port {} is no longer listening after cleanup",
                port
            ));
            return;
        }
        sleep(Duration::from_millis(250)).await;
    }

    log_info(format!(
        "SGLang stale-shutdown check: port {} is still listening after pkill cleanup",
        port
    ));
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

pub fn resolve_sglang_port() -> u16 {
    std::env::var("SGLANG_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port > 0)
        .unwrap_or(30000)
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

async fn wait_for_sglang_ready(port: u16, process: &mut Child) -> Result<(), String> {
    let timeout_duration = Duration::from_secs(180);
    let sleep_interval = Duration::from_millis(500);
    let start = Instant::now();
    let address = SocketAddr::from(([127, 0, 0, 1], port));

    loop {
        if timeout(
            Duration::from_millis(250),
            tokio::net::TcpStream::connect(address),
        )
        .await
        .is_ok_and(|result| result.is_ok())
        {
            return Ok(());
        }

        match process.try_wait() {
            Ok(Some(status)) => {
                return Err(format!(
                    "Sglang process exited before becoming ready (status: {}).",
                    status
                ));
            }
            Ok(None) => {}
            Err(err) => {
                return Err(format!(
                    "Failed to check sglang process status while waiting: {}",
                    err
                ));
            }
        }

        if start.elapsed() >= timeout_duration {
            return Err(format!(
                "Timed out waiting for sglang to listen on port {} after {} seconds",
                port,
                timeout_duration.as_secs()
            ));
        }
        sleep(sleep_interval).await;
    }
}
