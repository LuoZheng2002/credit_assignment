use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use research_utility::{
    message::TuiMessage,
    progress_tui_logger::{log_info, log_message, log_warning},
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::time::{Instant, sleep, timeout};

use crate::launch_sglang_server::resolve_sglang_port;

static WRAPPER_TUI_SOCKET_COUNTER: AtomicU64 = AtomicU64::new(0);
const WRAPPER_TUI_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const WRAPPER_TUI_ADDITIONAL_CONNECT_IDLE_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn launch_inference_wrapper_process(
    model_path: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
    hf_model_name: &str,
    num_gpus: usize,
    wrapper_log_path: &str,
) -> Result<(u16, Child), String> {
    let listen_port = resolve_sglang_port();
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
        .arg("--wrapper-log-path")
        .arg(wrapper_log_path)
        .arg("--orchestrator-socket-path")
        .arg(&socket_path_arg)
        .arg("--hf-model-name")
        .arg(hf_model_name)
        .arg("--model-path")
        .arg(model_path);

    #[cfg(unix)]
    command.process_group(0);

    command.stdout(Stdio::null());
    command.stderr(Stdio::null());

    let mut process = match command.spawn() {
        Ok(process) => process,
        Err(err) => {
            cleanup_socket_path(&socket_path);
            return Err(format!(
                "failed to launch inference wrapper process: {}",
                err
            ));
        }
    };

    spawn_wrapper_tui_listener(listener, socket_path, "inference wrapper", false);

    wait_for_wrapper_health(listen_port, &mut process, wrapper_log_path).await?;
    Ok((listen_port, process))
}

pub async fn run_training_wrapper_and_wait(
    num_gpus: usize,
    hf_model_name: &str,
    training_config_json: String,
    trajectory_sqlite_path: &str,
    wrapper_log_path: &str,
) -> Result<(), String> {
    assert!(num_gpus > 0, "num_gpus must be positive");
    if !Path::new(trajectory_sqlite_path).is_file() {
        return Err(format!(
            "training trajectory sqlite path does not exist: {}",
            trajectory_sqlite_path
        ));
    }
    let (socket_path, listener) = bind_wrapper_tui_listener("training")?;
    let socket_path_arg = socket_path_to_arg(&socket_path)?;

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("-m")
        .arg("src_py.wrappers.training_wrapper")
        .arg("--num-gpus")
        .arg(num_gpus.to_string())
        .arg("--training-config-json")
        .arg(training_config_json)
        .arg("--trajectory-sqlite-path")
        .arg(trajectory_sqlite_path)
        .arg("--hf-model-name")
        .arg(hf_model_name)
        .arg("--wrapper-log-path")
        .arg(wrapper_log_path)
        .arg("--orchestrator-socket-path")
        .arg(&socket_path_arg);

    command.stdout(Stdio::null());
    command.stderr(Stdio::null());

    let mut process = match command.spawn() {
        Ok(process) => process,
        Err(err) => {
            cleanup_socket_path(&socket_path);
            return Err(format!(
                "failed to launch training wrapper process: {}",
                err
            ));
        }
    };

    spawn_wrapper_tui_listener(listener, socket_path, "training wrapper", true);

    let status = process
        .wait()
        .await
        .map_err(|err| format!("failed while waiting for training wrapper process: {}", err))?;

    if status.success() {
        log_info(format!(
            "Training wrapper completed successfully; details in {}",
            wrapper_log_path
        ));
        Ok(())
    } else {
        Err(format!(
            "training wrapper process exited with status {}; inspect log at {}",
            status, wrapper_log_path
        ))
    }
}

fn bind_wrapper_tui_listener(wrapper_kind: &str) -> Result<(PathBuf, UnixListener), String> {
    let socket_path = wrapper_tui_socket_path(wrapper_kind);
    cleanup_socket_path(&socket_path);
    let listener = UnixListener::bind(&socket_path).map_err(|err| {
        format!(
            "failed to bind Unix socket for {} wrapper TUI listener at {}: {}",
            wrapper_kind,
            socket_path.display(),
            err
        )
    })?;
    Ok((socket_path, listener))
}

fn spawn_wrapper_tui_listener(
    listener: UnixListener,
    socket_path: PathBuf,
    wrapper_name: &'static str,
    allow_multiple_connections: bool,
) {
    tokio::spawn(async move {
        if let Err(err) = run_wrapper_tui_listener(
            listener,
            &socket_path,
            wrapper_name,
            allow_multiple_connections,
        )
        .await
        {
            log_warning(format!(
                "{} TUI socket listener ended with error: {}",
                wrapper_name, err
            ));
        }
        cleanup_socket_path(&socket_path);
    });
}

async fn run_wrapper_tui_listener(
    listener: UnixListener,
    socket_path: &Path,
    wrapper_name: &str,
    allow_multiple_connections: bool,
) -> Result<(), String> {
    let mut connection_tasks = Vec::new();
    let (first_stream, _) = timeout(WRAPPER_TUI_CONNECT_TIMEOUT, listener.accept())
        .await
        .map_err(|_| {
            format!(
                "timed out waiting for {} to connect to Unix socket {}",
                wrapper_name,
                socket_path.display()
            )
        })?
        .map_err(|err| {
            format!(
                "failed while accepting {} Unix socket connection at {}: {}",
                wrapper_name,
                socket_path.display(),
                err
            )
        })?;
    connection_tasks.push(tokio::spawn(read_tui_stream(
        first_stream,
        wrapper_name.to_string(),
    )));

    if allow_multiple_connections {
        loop {
            match timeout(
                WRAPPER_TUI_ADDITIONAL_CONNECT_IDLE_TIMEOUT,
                listener.accept(),
            )
            .await
            {
                Ok(Ok((stream, _))) => {
                    connection_tasks.push(tokio::spawn(read_tui_stream(
                        stream,
                        wrapper_name.to_string(),
                    )));
                }
                Ok(Err(err)) => {
                    return Err(format!(
                        "failed while accepting additional {} Unix socket connection at {}: {}",
                        wrapper_name,
                        socket_path.display(),
                        err
                    ));
                }
                Err(_) => break,
            }
        }
    }

    for task in connection_tasks {
        match task.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err),
            Err(err) => {
                return Err(format!(
                    "{} TUI socket reader task join failure: {}",
                    wrapper_name, err
                ));
            }
        }
    }
    Ok(())
}

async fn read_tui_stream(stream: UnixStream, wrapper_name: String) -> Result<(), String> {
    let mut lines = BufReader::new(stream).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(content)) => {
                if content.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<TuiMessage>(&content) {
                    Ok(message) => log_message(message),
                    Err(err) => log_warning(format!(
                        "failed to parse {} TUI socket message as TuiMessage: {} (payload={})",
                        wrapper_name, err, content
                    )),
                }
            }
            Ok(None) => return Ok(()),
            Err(err) => {
                return Err(format!(
                    "failed while reading {} TUI socket stream: {}",
                    wrapper_name, err
                ));
            }
        }
    }
}

fn socket_path_to_arg(socket_path: &Path) -> Result<String, String> {
    socket_path
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("socket path is not valid UTF-8: {}", socket_path.display()))
}

fn wrapper_tui_socket_path(wrapper_kind: &str) -> PathBuf {
    let id = WRAPPER_TUI_SOCKET_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "credit_assignment_{}_wrapper_{}_{}.sock",
        wrapper_kind,
        std::process::id(),
        id
    ))
}

fn cleanup_socket_path(socket_path: &Path) {
    if let Err(err) = std::fs::remove_file(socket_path) {
        if err.kind() != std::io::ErrorKind::NotFound {
            log_warning(format!(
                "failed to remove Unix socket path {}: {}",
                socket_path.display(),
                err
            ));
        }
    }
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
            let hint = format!("; inspect wrapper log at {}", log_path);
            return Err(format!(
                "inference wrapper process exited before becoming healthy: {}{}",
                status, hint
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
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    use research_utility::{
        bincode_log_file::BincodeLogFile,
        message::{Severity, TuiMessage},
        progress_tui_logger::{ProgressLogFrame, ProgressTuiLogger},
    };
    use tokio::io::AsyncWriteExt;
    use tokio::net::UnixStream;

    use super::{bind_wrapper_tui_listener, run_wrapper_tui_listener};

    #[test]
    fn state_tui_message_json_parses() {
        let json = r#"{"State":{"state":"Inference wrapper started"}}"#;
        let parsed = serde_json::from_str::<TuiMessage>(json).expect("state message should parse");
        match parsed {
            TuiMessage::State { state } => assert_eq!(state, "Inference wrapper started"),
            _ => panic!("expected state message"),
        }
    }

    #[test]
    fn line_tui_message_json_parses() {
        let json = r#"{"Line":{"message":"Training failed: torchrun exited with code 1","severity":"Error"}}"#;
        let parsed = serde_json::from_str::<TuiMessage>(json).expect("line message should parse");
        match parsed {
            TuiMessage::Line { message, severity } => {
                assert_eq!(message, "Training failed: torchrun exited with code 1");
                assert_eq!(severity, Severity::Error);
            }
            _ => panic!("expected line message"),
        }
    }

    #[test]
    fn key_value_tui_message_json_parses() {
        let json = r#"{"KeyValuePair":{"key":"checkpoint_dir","value":"/tmp/checkpoints"}}"#;
        let parsed =
            serde_json::from_str::<TuiMessage>(json).expect("key value message should parse");
        match parsed {
            TuiMessage::KeyValuePair { key, value } => {
                assert_eq!(key, "checkpoint_dir");
                assert_eq!(value, "/tmp/checkpoints");
            }
            _ => panic!("expected key value message"),
        }
    }

    #[tokio::test]
    async fn unix_socket_tui_listener_forwards_messages_into_progress_log() {
        let log_path = temp_progress_log_path("wrapper_socket_forward");
        ProgressTuiLogger::initialize(&log_path)
            .await
            .expect("progress logger should initialize");

        let (socket_path, listener) =
            bind_wrapper_tui_listener("test").expect("listener should bind");
        let socket_path_for_listener = socket_path.clone();
        let listener_task = tokio::spawn(async move {
            run_wrapper_tui_listener(listener, &socket_path_for_listener, "test wrapper", true)
                .await
                .expect("listener should complete successfully");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("first client should connect");
        stream
            .write_all(b"{\"State\":{\"state\":\"Training wrapper started\"}}\n")
            .await
            .expect("state message should write");
        drop(stream);

        let mut second_stream = UnixStream::connect(&socket_path)
            .await
            .expect("second client should connect");
        second_stream
            .write_all(
                b"{\"KeyValuePair\":{\"key\":\"checkpoint_dir\",\"value\":\"/tmp/checkpoints\"}}\n",
            )
            .await
            .expect("key value message should write");
        second_stream
            .write_all(
                b"{\"Line\":{\"message\":\"Training failed: torchrun exited with code 1\",\"severity\":\"Error\"}}\n",
            )
            .await
            .expect("line message should write");
        drop(second_stream);

        listener_task
            .await
            .expect("listener task should join successfully");
        ProgressTuiLogger::shutdown()
            .await
            .expect("progress logger should shutdown cleanly");

        let frames = read_all_progress_frames(&log_path);
        assert!(
            !frames.is_empty(),
            "expected at least one progress log frame after socket forwarding"
        );
        assert!(
            frames
                .iter()
                .any(|frame| frame.state.as_deref() == Some("Training wrapper started")),
            "expected forwarded state message in progress frames"
        );
        assert!(
            frames.iter().any(|frame| {
                frame
                    .key_values
                    .iter()
                    .any(|(key, value)| key == "checkpoint_dir" && value == "/tmp/checkpoints")
            }),
            "expected forwarded key/value message in progress frames"
        );
        assert!(
            frames.iter().any(|frame| {
                frame.log_lines.iter().any(|line| {
                    line.message == "Training failed: torchrun exited with code 1"
                        && line.severity == Severity::Error
                })
            }),
            "expected forwarded log line in progress frames"
        );

        let _ = std::fs::remove_file(&log_path);
    }

    fn read_all_progress_frames(path: &Path) -> Vec<ProgressLogFrame> {
        let log_file = BincodeLogFile::<ProgressLogFrame>::open(path)
            .expect("progress log file should open for reading");
        log_file
            .iter()
            .expect("progress log iterator should open")
            .map(|frame| frame.expect("progress log frame should deserialize"))
            .collect()
    }

    fn temp_progress_log_path(label: &str) -> std::path::PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "credit_assignment_progress_tui_{}_{}_{}.bin",
            label,
            std::process::id(),
            now
        ))
    }
}
