use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
};

use research_utility::{
    message::TuiMessage,
    progress_tui_logger::{log_message, log_warning},
};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::watch;

static WRAPPER_TUI_SOCKET_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn bind_wrapper_tui_listener(
    wrapper_kind: &str,
) -> Result<(PathBuf, UnixListener), String> {
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

pub(crate) fn socket_path_to_arg(socket_path: &Path) -> Result<String, String> {
    socket_path
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("socket path is not valid UTF-8: {}", socket_path.display()))
}

pub(crate) fn spawn_wrapper_command(
    command: &mut Command,
    socket_path: &Path,
    wrapper_name: &str,
) -> Result<Child, String> {
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());
    command.spawn().map_err(|err| {
        cleanup_socket_path(socket_path);
        format!("failed to launch {} process: {}", wrapper_name, err)
    })
}

pub(crate) async fn write_json_payload_to_child_stdin<T: Serialize>(
    child: &mut Child,
    payload: &T,
    process_name: &str,
) -> Result<(), String> {
    let stdin_payload = serde_json::to_vec(payload).map_err(|err| {
        format!(
            "failed to serialize {} stdin payload as JSON: {}",
            process_name, err
        )
    })?;
    let mut stdin = child.stdin.take().ok_or_else(|| {
        format!(
            "{} stdin is unavailable; expected a piped stdin handle",
            process_name
        )
    })?;
    stdin.write_all(&stdin_payload).await.map_err(|err| {
        format!(
            "failed to write JSON stdin payload to {}: {}",
            process_name, err
        )
    })?;
    stdin.shutdown().await.map_err(|err| {
        format!(
            "failed to close {} stdin after writing JSON payload: {}",
            process_name, err
        )
    })?;
    drop(stdin);
    Ok(())
}

pub(crate) fn spawn_wrapper_tui_listener(
    listener: UnixListener,
    socket_path: PathBuf,
    wrapper_name: &'static str,
    allow_multiple_connections: bool,
    stop_signal: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(err) = run_wrapper_tui_listener(
            listener,
            &socket_path,
            wrapper_name,
            allow_multiple_connections,
            stop_signal,
        )
        .await
        {
            log_warning(format!(
                "{} TUI socket listener ended with error: {}",
                wrapper_name, err
            ));
        }
        cleanup_socket_path(&socket_path);
    })
}

async fn run_wrapper_tui_listener(
    listener: UnixListener,
    socket_path: &Path,
    wrapper_name: &str,
    allow_multiple_connections: bool,
    mut stop_signal: watch::Receiver<bool>,
) -> Result<(), String> {
    if *stop_signal.borrow() {
        return Ok(());
    }
    let (first_stream, _) = tokio::select! {
        accept_result = listener.accept() => {
            accept_result.map_err(|err| {
                format!(
                    "failed while accepting {} Unix socket connection at {}: {}",
                    wrapper_name,
                    socket_path.display(),
                    err
                )
            })?
        }
        stop_result = stop_signal.changed() => {
            let _ = stop_result;
            return Ok(());
        }
    };

    let primary_connection_name = wrapper_name.to_string();
    let mut primary_connection_task =
        tokio::spawn(read_tui_stream(first_stream, primary_connection_name));
    let mut additional_connection_tasks: Vec<tokio::task::JoinHandle<Result<(), String>>> =
        Vec::new();

    if allow_multiple_connections {
        loop {
            if *stop_signal.borrow() {
                primary_connection_task.abort();
                let _ = primary_connection_task.await;
                for task in additional_connection_tasks {
                    task.abort();
                    let _ = task.await;
                }
                return Ok(());
            }
            tokio::select! {
                primary_result = &mut primary_connection_task => {
                    match primary_result {
                        Ok(Ok(())) => break,
                        Ok(Err(err)) => return Err(err),
                        Err(err) => {
                            return Err(format!(
                                "{} primary TUI socket reader task join failure: {}",
                                wrapper_name, err
                            ));
                        }
                    }
                }
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, _)) => {
                            additional_connection_tasks.push(tokio::spawn(read_tui_stream(
                                stream,
                                wrapper_name.to_string(),
                            )));
                        }
                        Err(err) => {
                            return Err(format!(
                                "failed while accepting additional {} Unix socket connection at {}: {}",
                                wrapper_name,
                                socket_path.display(),
                                err
                            ));
                        }
                    }
                }
                stop_result = stop_signal.changed() => {
                    let _ = stop_result;
                    primary_connection_task.abort();
                    let _ = primary_connection_task.await;
                    for task in additional_connection_tasks {
                        task.abort();
                        let _ = task.await;
                    }
                    return Ok(());
                }
            }
        }
    } else {
        tokio::select! {
            primary_result = &mut primary_connection_task => {
                match primary_result {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(err),
                    Err(err) => {
                        return Err(format!(
                            "{} primary TUI socket reader task join failure: {}",
                            wrapper_name, err
                        ));
                    }
                }
            }
            stop_result = stop_signal.changed() => {
                let _ = stop_result;
                primary_connection_task.abort();
                let _ = primary_connection_task.await;
                return Ok(());
            }
        }
    }

    for task in additional_connection_tasks {
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

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::OnceLock;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use research_utility::{
        bincode_log_file::BincodeLogFile,
        message::{Severity, TuiMessage},
        progress_tui_logger::{ProgressLogFrame, ProgressTuiLogger},
    };
    use tokio::io::AsyncWriteExt;
    use tokio::net::UnixStream;
    use tokio::sync::watch;

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
        let _guard = progress_logger_test_lock().lock().await;
        let log_path = temp_progress_log_path("wrapper_socket_forward");
        ProgressTuiLogger::initialize(&log_path)
            .await
            .expect("progress logger should initialize");

        let (socket_path, listener) =
            bind_wrapper_tui_listener("test").expect("listener should bind");
        let socket_path_for_listener = socket_path.clone();
        let (_stop_signal_tx, stop_signal_rx) = watch::channel(false);
        let listener_task = tokio::spawn(async move {
            run_wrapper_tui_listener(
                listener,
                &socket_path_for_listener,
                "test wrapper",
                true,
                stop_signal_rx,
            )
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

    #[tokio::test]
    async fn unix_socket_tui_listener_accepts_delayed_second_connection_while_primary_is_open() {
        let _guard = progress_logger_test_lock().lock().await;
        let log_path = temp_progress_log_path("wrapper_socket_delayed_second_connection");
        ProgressTuiLogger::initialize(&log_path)
            .await
            .expect("progress logger should initialize");

        let (socket_path, listener) =
            bind_wrapper_tui_listener("test").expect("listener should bind");
        let socket_path_for_listener = socket_path.clone();
        let (_stop_signal_tx, stop_signal_rx) = watch::channel(false);
        let listener_task = tokio::spawn(async move {
            run_wrapper_tui_listener(
                listener,
                &socket_path_for_listener,
                "test wrapper",
                true,
                stop_signal_rx,
            )
            .await
            .expect("listener should complete successfully");
        });

        let mut primary_stream = UnixStream::connect(&socket_path)
            .await
            .expect("primary client should connect");
        primary_stream
            .write_all(b"{\"State\":{\"state\":\"Training wrapper started\"}}\n")
            .await
            .expect("state message should write");

        tokio::time::sleep(Duration::from_secs(6)).await;

        let mut second_stream = UnixStream::connect(&socket_path)
            .await
            .expect("delayed second client should connect");
        second_stream
            .write_all(
                b"{\"Line\":{\"message\":\"delayed training info\",\"severity\":\"Info\"}}\n",
            )
            .await
            .expect("delayed line message should write");
        drop(second_stream);
        drop(primary_stream);

        listener_task
            .await
            .expect("listener task should join successfully");
        ProgressTuiLogger::shutdown()
            .await
            .expect("progress logger should shutdown cleanly");

        let frames = read_all_progress_frames(&log_path);
        assert!(
            frames.iter().any(|frame| {
                frame.log_lines.iter().any(|line| {
                    line.message == "delayed training info" && line.severity == Severity::Info
                })
            }),
            "expected delayed second connection log line in progress frames"
        );

        let _ = std::fs::remove_file(&log_path);
    }

    fn progress_logger_test_lock() -> &'static tokio::sync::Mutex<()> {
        static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
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
