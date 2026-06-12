use std::{collections::VecDeque, process::Stdio, sync::Arc, time::Duration};

use research_utility::progress_tui_logger::log_warning;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    sync::{Mutex, mpsc},
    task::JoinHandle,
    time::timeout,
};

use crate::{
    llm_model::{LlmModelMarker, MyTokenizer},
    token_array::TokenArray,
};

pub fn extract_python_tool_call(response: String) -> Option<String> {
    let python_fence = "```python";
    let Some(python_start_position) = response.find(python_fence) else {
        return None;
    };

    let start_position = python_start_position;
    let end_position = {
        let search_start = python_start_position + python_fence.len();
        let after_python_fence = &response[search_start..];
        if let Some(end_relative) = after_python_fence.find("```\n") {
            search_start + end_relative + "```\n".len()
        } else if let Some(end_relative) = after_python_fence.find("```") {
            let mut end_position = search_start + end_relative + "```".len();
            if after_python_fence[end_relative + "```".len()..].starts_with('\n') {
                end_position += 1;
            }
            end_position
        } else {
            return None;
        }
    };
    Some(response[start_position..end_position].to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PythonToolResponse {
    PythonSuccess(String),
    PythonError(String),
}

impl PythonToolResponse {
    pub fn with_multi_turn_chat_template<M: LlmModelMarker>(
        &self,
        enable_thinking: bool,
    ) -> TokenArray<M> {
        let raw_python_response = match self {
            PythonToolResponse::PythonSuccess(output) => output.clone(),
            PythonToolResponse::PythonError(error) => format!("Python error: {}", error),
        };
        M::Tokenizer::apply_python_response_template_and_tokenize(
            raw_python_response,
            enable_thinking,
        )
    }
}

const PYTHON_TOOL_REQUEST_TIMEOUT_MS: u64 = 1000;
const PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS: u64 = 3000;
const PYTHON_TOOL_SERVER_STARTUP_TIMEOUT_MS: u64 = 10_000;
const PYTHON_TOOL_PROCESS_SHUTDOWN_TIMEOUT_MS: u64 = 1000;
const PYTHON_TOOL_STDERR_DRAIN_TIMEOUT_MS: u64 = 1000;
const PYTHON_TOOL_STDERR_TAIL_LINES: usize = 40;

fn python_request_timeout_error_message() -> String {
    format!(
        "Python code execution timed out after {} ms.",
        PYTHON_TOOL_REQUEST_TIMEOUT_MS
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonToolReadyWire {
    ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonToolRequestWire {
    code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonToolResponseWire {
    ok: bool,
    output: Option<String>,
    error: Option<String>,
}

struct PythonToolServerWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    stderr_task: JoinHandle<()>,
}

struct PythonToolServerSlot {
    worker: Option<PythonToolServerWorker>,
}

pub struct PythonToolServerPool {
    slots: Vec<Arc<Mutex<PythonToolServerSlot>>>,
    available_sender: mpsc::UnboundedSender<usize>,
    available_receiver: Mutex<mpsc::UnboundedReceiver<usize>>,
}

impl PythonToolServerPool {
    pub async fn new(max_python_processes: usize) -> Result<Self, String> {
        if max_python_processes == 0 {
            return Err("max_python_processes must be greater than zero".to_string());
        }

        let (available_sender, available_receiver) = mpsc::unbounded_channel();
        let mut slots = Vec::with_capacity(max_python_processes);
        for worker_id in 0..max_python_processes {
            let worker = spawn_python_tool_server_worker().await.map_err(|error| {
                format!(
                    "Failed to initialize python tool server worker {}: {}",
                    worker_id, error
                )
            })?;
            slots.push(Arc::new(Mutex::new(PythonToolServerSlot {
                worker: Some(worker),
            })));
            available_sender
                .send(worker_id)
                .map_err(|_| "failed to seed python tool worker availability queue".to_string())?;
        }

        Ok(Self {
            slots,
            available_sender,
            available_receiver: Mutex::new(available_receiver),
        })
    }

    pub async fn execute_code(&self, code: String) -> PythonToolResponse {
        let worker_id = {
            let mut receiver = self.available_receiver.lock().await;
            match receiver.recv().await {
                Some(worker_id) => worker_id,
                None => {
                    return PythonToolResponse::PythonError(
                        "Python tool worker queue closed unexpectedly.".to_string(),
                    );
                }
            }
        };

        let response = {
            let slot = self
                .slots
                .get(worker_id)
                .expect("worker id from queue must refer to an existing slot");
            let mut slot_guard = slot.lock().await;

            if slot_guard.worker.is_none() {
                if let Err(error) = ensure_worker_running(&mut slot_guard).await {
                    PythonToolResponse::PythonError(error)
                } else {
                    execute_request_with_slot(&mut slot_guard, worker_id, code).await
                }
            } else {
                execute_request_with_slot(&mut slot_guard, worker_id, code).await
            }
        };

        let _ = self.available_sender.send(worker_id);
        response
    }
}

impl PythonToolServerWorker {
    async fn execute_request(&mut self, code: String) -> Result<PythonToolResponse, String> {
        let request = serde_json::to_string(&PythonToolRequestWire { code })
            .map_err(|error| format!("Failed to serialize python tool request: {}", error))?;

        if let Err(error) = self.stdin.write_all(request.as_bytes()).await {
            return Err(format!(
                "Failed to write request to python tool server stdin: {}{}",
                error,
                self.stderr_context().await
            ));
        }
        if let Err(error) = self.stdin.write_all(b"\n").await {
            return Err(format!(
                "Failed to finalize request line to python tool server stdin: {}{}",
                error,
                self.stderr_context().await
            ));
        }

        let mut response_line = String::new();
        let bytes_read = match self.stdout.read_line(&mut response_line).await {
            Ok(bytes_read) => bytes_read,
            Err(error) => {
                return Err(format!(
                    "Failed to read response from python tool server stdout: {}{}",
                    error,
                    self.stderr_context().await
                ));
            }
        };
        if bytes_read == 0 {
            return Err(format!(
                "Python tool server closed stdout before returning a response.{}",
                self.stderr_context().await
            ));
        }

        match parse_python_response_line(&response_line) {
            Ok(response) => Ok(response),
            Err(error) => Err(format!("{}{}", error, self.stderr_context().await)),
        }
    }

    async fn stderr_context(&self) -> String {
        let stderr_tail = self.stderr_tail.lock().await;
        if stderr_tail.is_empty() {
            String::new()
        } else {
            format!(
                " stderr_tail={}",
                stderr_tail.iter().cloned().collect::<Vec<_>>().join(" | ")
            )
        }
    }

    async fn terminate(&mut self) {
        let child_pid = self.child.id();
        terminate_tool_process(&mut self.child, child_pid).await;
        await_stderr_task(&mut self.stderr_task).await;
    }
}

async fn execute_request_with_slot(
    slot: &mut PythonToolServerSlot,
    worker_id: usize,
    code: String,
) -> PythonToolResponse {
    let request_result = {
        let worker = slot
            .worker
            .as_mut()
            .expect("worker must exist before executing a request");
        timeout(
            Duration::from_millis(PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS),
            worker.execute_request(code),
        )
        .await
    };

    match request_result {
        Ok(Ok(raw_response)) => normalize_python_response(raw_response),
        Ok(Err(error)) => {
            log_warning(format!(
                "Python tool server worker {} failed while handling a request: {}",
                worker_id, error
            ));
            let restart_note = restart_worker(slot, worker_id).await;
            PythonToolResponse::PythonError(format!(
                "Python tool server request failed: {}{}",
                error, restart_note
            ))
        }
        Err(_) => {
            let restart_note = restart_worker(slot, worker_id).await;
            log_warning(format!(
                "Python tool server worker {} did not respond within {} ms; returning a normalized request-timeout error to the model and restarting worker.{}",
                worker_id, PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS, restart_note
            ));
            PythonToolResponse::PythonError(python_request_timeout_error_message())
        }
    }
}

async fn ensure_worker_running(slot: &mut PythonToolServerSlot) -> Result<(), String> {
    if slot.worker.is_some() {
        return Ok(());
    }
    slot.worker = Some(spawn_python_tool_server_worker().await?);
    Ok(())
}

async fn restart_worker(slot: &mut PythonToolServerSlot, worker_id: usize) -> String {
    if let Some(mut worker) = slot.worker.take() {
        worker.terminate().await;
    }
    match spawn_python_tool_server_worker().await {
        Ok(worker) => {
            slot.worker = Some(worker);
            String::new()
        }
        Err(error) => {
            log_warning(format!(
                "Failed to respawn python tool server worker {}: {}",
                worker_id, error
            ));
            format!(" Failed to respawn replacement worker: {}", error)
        }
    }
}

fn normalize_python_response(raw_response: PythonToolResponse) -> PythonToolResponse {
    match raw_response {
        PythonToolResponse::PythonSuccess(output) => {
            if output.trim().is_empty() {
                log_warning("Python interpreter did not return any output.");
                PythonToolResponse::PythonSuccess(
                    "Python interpreter did not return any output. Please use print statements to retrieve results.".to_string(),
                )
            } else {
                PythonToolResponse::PythonSuccess(output)
            }
        }
        PythonToolResponse::PythonError(error) => PythonToolResponse::PythonError(error),
    }
}

async fn spawn_python_tool_server_worker() -> Result<PythonToolServerWorker, String> {
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("python")
        .arg("-u")
        .arg("-B")
        .arg("-m")
        .arg("src_py.tool_server.main")
        .arg("--persistent-server")
        .arg("--request-timeout-ms")
        .arg(PYTHON_TOOL_REQUEST_TIMEOUT_MS.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONUNBUFFERED", "1");

    #[cfg(unix)]
    command.process_group(0);

    let mut child = command.spawn().map_err(|error| {
        format!(
            "Failed to spawn persistent python tool server process: {}",
            error
        )
    })?;
    let child_pid = child.id();

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to capture persistent python tool server stdin".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture persistent python tool server stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Failed to capture persistent python tool server stderr".to_string())?;

    let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(
        PYTHON_TOOL_STDERR_TAIL_LINES,
    )));
    let mut stderr_task = tokio::spawn(drain_stderr(stderr, stderr_tail.clone()));
    let mut stdout = BufReader::new(stdout);

    let ready_result = timeout(
        Duration::from_millis(PYTHON_TOOL_SERVER_STARTUP_TIMEOUT_MS),
        read_json_line(&mut stdout),
    )
    .await;

    let ready_line = match ready_result {
        Ok(Ok(line)) => line,
        Ok(Err(error)) => {
            terminate_tool_process(&mut child, child_pid).await;
            await_stderr_task(&mut stderr_task).await;
            return Err(format!(
                "Python tool server failed during startup handshake: {}{}",
                error,
                stderr_context_from_tail(&stderr_tail).await
            ));
        }
        Err(_) => {
            terminate_tool_process(&mut child, child_pid).await;
            await_stderr_task(&mut stderr_task).await;
            return Err(format!(
                "Python tool server did not signal readiness within {} ms.{}",
                PYTHON_TOOL_SERVER_STARTUP_TIMEOUT_MS,
                stderr_context_from_tail(&stderr_tail).await
            ));
        }
    };

    match serde_json::from_str::<PythonToolReadyWire>(ready_line.trim()) {
        Ok(ready) if ready.ready => Ok(PythonToolServerWorker {
            child,
            stdin,
            stdout,
            stderr_tail,
            stderr_task,
        }),
        Ok(_) => {
            terminate_tool_process(&mut child, child_pid).await;
            await_stderr_task(&mut stderr_task).await;
            Err(format!(
                "Python tool server returned an invalid readiness payload.{}",
                stderr_context_from_tail(&stderr_tail).await
            ))
        }
        Err(error) => {
            terminate_tool_process(&mut child, child_pid).await;
            await_stderr_task(&mut stderr_task).await;
            Err(format!(
                "Failed to parse python tool server readiness payload: {}. Raw stdout: {}{}",
                error,
                ready_line.trim(),
                stderr_context_from_tail(&stderr_tail).await
            ))
        }
    }
}

async fn drain_stderr(stderr: ChildStderr, stderr_tail: Arc<Mutex<VecDeque<String>>>) {
    let mut reader = BufReader::new(stderr);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {
                let cleaned = line.trim().to_string();
                if cleaned.is_empty() {
                    continue;
                }
                let mut stderr_tail = stderr_tail.lock().await;
                if stderr_tail.len() == PYTHON_TOOL_STDERR_TAIL_LINES {
                    stderr_tail.pop_front();
                }
                stderr_tail.push_back(cleaned);
            }
            Err(error) => {
                let mut stderr_tail = stderr_tail.lock().await;
                if stderr_tail.len() == PYTHON_TOOL_STDERR_TAIL_LINES {
                    stderr_tail.pop_front();
                }
                stderr_tail.push_back(format!(
                    "Failed to read python tool server stderr: {}",
                    error
                ));
                break;
            }
        }
    }
}

async fn read_json_line(stdout: &mut BufReader<ChildStdout>) -> Result<String, String> {
    let mut line = String::new();
    let bytes_read = stdout.read_line(&mut line).await.map_err(|error| {
        format!(
            "Failed to read line from python tool server stdout: {}",
            error
        )
    })?;
    if bytes_read == 0 {
        return Err("Python tool server stdout closed unexpectedly".to_string());
    }
    Ok(line)
}

async fn stderr_context_from_tail(stderr_tail: &Arc<Mutex<VecDeque<String>>>) -> String {
    let stderr_tail = stderr_tail.lock().await;
    if stderr_tail.is_empty() {
        String::new()
    } else {
        format!(
            " stderr_tail={}",
            stderr_tail.iter().cloned().collect::<Vec<_>>().join(" | ")
        )
    }
}

async fn terminate_tool_process(child: &mut Child, child_pid: Option<u32>) {
    #[cfg(unix)]
    {
        if let Some(pid) = child_pid {
            let _ = Command::new("kill")
                .arg("-KILL")
                .arg("--")
                .arg(format!("-{}", pid))
                .status()
                .await;
        }
    }

    let _ = child.start_kill();
    let wait_result = timeout(
        Duration::from_millis(PYTHON_TOOL_PROCESS_SHUTDOWN_TIMEOUT_MS),
        child.wait(),
    )
    .await;
    if wait_result.is_err() {
        log_warning(format!(
            "Python tool process did not exit within {} ms after kill signal (pid={:?}).",
            PYTHON_TOOL_PROCESS_SHUTDOWN_TIMEOUT_MS, child_pid
        ));
    }
}

async fn await_stderr_task(stderr_task: &mut JoinHandle<()>) {
    let join_result = timeout(
        Duration::from_millis(PYTHON_TOOL_STDERR_DRAIN_TIMEOUT_MS),
        async {
            let _ = (&mut *stderr_task).await;
        },
    )
    .await;
    if join_result.is_err() {
        stderr_task.abort();
        log_warning(format!(
            "Timed out draining python tool server stderr within {} ms.",
            PYTHON_TOOL_STDERR_DRAIN_TIMEOUT_MS
        ));
    }
}

fn parse_python_response_line(response_line: &str) -> Result<PythonToolResponse, String> {
    let trimmed_stdout = response_line.trim();
    if trimmed_stdout.is_empty() {
        return Err("Python tool server returned an empty response line.".to_string());
    }

    match serde_json::from_str::<PythonToolResponseWire>(trimmed_stdout) {
        Ok(response_wire) => {
            if response_wire.ok {
                Ok(PythonToolResponse::PythonSuccess(
                    response_wire.output.unwrap_or_default(),
                ))
            } else {
                Ok(PythonToolResponse::PythonError(
                    response_wire.error.unwrap_or_else(|| {
                        "Unknown python interpreter execution error".to_string()
                    }),
                ))
            }
        }
        Err(error) => Err(format!(
            "Failed to parse python tool server response as JSON: {}. Raw stdout: {}",
            error, trimmed_stdout
        )),
    }
}

pub async fn execute_python_tool_call(
    pool: &PythonToolServerPool,
    tool_call: &str,
) -> PythonToolResponse {
    let trimmed_tool_call = tool_call.trim_start().to_string();
    assert!(
        trimmed_tool_call.starts_with("```python"),
        "Tool call not properly formatted: {}",
        tool_call
    );
    let code_start = trimmed_tool_call
        .find('\n')
        .map(|idx| idx + 1)
        .unwrap_or("```python".len());
    let fence_end_index = trimmed_tool_call[code_start..]
        .find("```")
        .map(|relative_idx| code_start + relative_idx);
    let Some(fence_end_index) = fence_end_index else {
        return PythonToolResponse::PythonError(
            "Tool call markdown code block not properly closed.".to_string(),
        );
    };
    if fence_end_index < code_start {
        return PythonToolResponse::PythonError(
            "Tool call markdown code block not properly formatted.".to_string(),
        );
    }
    let code = &trimmed_tool_call[code_start..fence_end_index];
    pool.execute_code(code.to_string()).await
}

#[cfg(test)]
mod tests {
    use super::{
        PYTHON_TOOL_REQUEST_TIMEOUT_MS, PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS, PythonToolResponse,
        PythonToolServerPool,
    };
    use tokio::time::{Duration, Instant, timeout};

    #[tokio::test]
    async fn python_tool_executes_simple_code() {
        let pool = PythonToolServerPool::new(1).await.unwrap();
        let response = pool.execute_code("print(1 + 1)".to_string()).await;
        assert_eq!(
            response,
            PythonToolResponse::PythonSuccess("2\n".to_string())
        );
    }

    #[tokio::test]
    async fn python_tool_state_does_not_persist_across_requests() {
        let pool = PythonToolServerPool::new(1).await.unwrap();
        let first = pool
            .execute_code("x = 41\nprint(\"set\")".to_string())
            .await;
        assert_eq!(
            first,
            PythonToolResponse::PythonSuccess("set\n".to_string())
        );

        let second = pool.execute_code("print(x)".to_string()).await;
        match second {
            PythonToolResponse::PythonError(message) => {
                assert!(
                    message.contains("name 'x' is not defined") || message.contains("NameError"),
                    "unexpected error message: {}",
                    message
                );
            }
            other => panic!("expected state-isolation error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn python_tool_blocks_writing_files_to_disk() {
        let pool = PythonToolServerPool::new(1).await.unwrap();
        let response = pool
            .execute_code(
                "with open('blocked.txt', 'w') as handle:\n    handle.write('x')".to_string(),
            )
            .await;
        match response {
            PythonToolResponse::PythonError(message) => {
                assert!(
                    message.contains("sandbox") || message.contains("forbids"),
                    "unexpected error message: {}",
                    message
                );
            }
            other => panic!("expected sandbox violation, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn python_tool_timeout_returns_promptly_and_pool_stays_usable() {
        let pool = PythonToolServerPool::new(1).await.unwrap();
        let code = r#"
import time
while True:
    pass
"#;
        let started = Instant::now();
        let response = timeout(
            Duration::from_millis(PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS + 2000),
            pool.execute_code(code.to_string()),
        )
        .await
        .expect("tool execution should return promptly after timeout handling");
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(PYTHON_TOOL_SERVER_RESPONSE_TIMEOUT_MS + 1000),
            "tool execution took too long: {:?}",
            elapsed
        );
        match response {
            PythonToolResponse::PythonError(message) => {
                assert!(
                    message.contains(&format!(
                        "timed out after {} ms",
                        PYTHON_TOOL_REQUEST_TIMEOUT_MS
                    )),
                    "unexpected error message: {}",
                    message
                );
            }
            other => panic!("expected timeout error, got {:?}", other),
        }

        let recovery = pool.execute_code("print(6 * 7)".to_string()).await;
        assert_eq!(
            recovery,
            PythonToolResponse::PythonSuccess("42\n".to_string())
        );
    }
}
