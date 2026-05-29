use std::{
    fs::OpenOptions,
    net::SocketAddr,
    process::{Child, Command, Stdio},
    time::Duration,
};

use tokio::time::{Instant, sleep, timeout};

use crate::{constants::SGLANG_CONTEXT_LENGTH, llm_model::LlmModelMarker};

pub fn model_uses_sglang<M: LlmModelMarker>() -> bool {
    !M::CLI_NAME.starts_with("gpt-")
}

pub async fn launch_sglang_server_process<M: LlmModelMarker>(
    model_path: &str,
    sglang_server_log_path: Option<&str>,
) -> (u16, Child) {
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
        .arg("--context-length")
        .arg(SGLANG_CONTEXT_LENGTH.to_string());

    if let Some(log_path) = sglang_server_log_path {
        let stdout_file = OpenOptions::new()
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

    wait_for_sglang_ready(sglang_port, &mut process).await;
    (sglang_port, process)
}

pub fn shut_down_sglang_server_process(process: &mut Child) {
    match process.try_wait() {
        Ok(Some(_)) => {}
        Ok(None) => {
            if let Err(err) = process.kill() {
                panic!("Failed to kill inference server process: {}", err);
            }
        }
        Err(err) => panic!("Failed to inspect inference server process status: {}", err),
    }

    if let Err(err) = process.wait() {
        panic!("Failed to wait on inference server process: {}", err);
    }
}

fn resolve_sglang_port() -> u16 {
    std::env::var("SGLANG_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port > 0)
        .unwrap_or(30000)
}

async fn wait_for_sglang_ready(port: u16, process: &mut Child) {
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
            return;
        }

        match process.try_wait() {
            Ok(Some(status)) => {
                panic!(
                    "Sglang process exited before becoming ready (status: {}).",
                    status
                );
            }
            Ok(None) => {}
            Err(err) => {
                panic!(
                    "Failed to check sglang process status while waiting: {}",
                    err
                );
            }
        }

        if start.elapsed() >= timeout_duration {
            panic!(
                "Timed out waiting for sglang to listen on port {} after {} seconds",
                port,
                timeout_duration.as_secs()
            );
        }
        sleep(sleep_interval).await;
    }
}
