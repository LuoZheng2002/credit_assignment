use std::{
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use research_utility::progress_tui_server::log_warning;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    sync::{mpsc, oneshot},
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
    // EmptyMessageHint,
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

const PYTHON_TOOL_TIMEOUT_MS: u64 = 5000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonToolRequestWire {
    id: u64,
    code: String,
    timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonToolResponseWire {
    id: u64,
    ok: bool,
    output: Option<String>,
    error: Option<String>,
}

struct ExecuteRequest {
    code: String,
    response_tx: oneshot::Sender<PythonToolResponse>,
}

struct PythonToolWorkerRuntime {
    child: Child,
    stdin: ChildStdin,
    stdout_lines: tokio::io::Lines<BufReader<ChildStdout>>,
    stderr_task: JoinHandle<()>,
}

pub struct PythonToolWorkerHandle {
    request_tx: mpsc::Sender<ExecuteRequest>,
    inflight: AtomicUsize,
}

impl PythonToolWorkerHandle {
    async fn spawn(worker_id: usize, parent_pid: u32) -> Result<Self, String> {
        let (request_tx, mut request_rx) = mpsc::channel::<ExecuteRequest>(1024);
        tokio::spawn(async move {
            let mut runtime: Option<PythonToolWorkerRuntime> = None;
            let mut next_request_id: u64 = 1;
            while let Some(request) = request_rx.recv().await {
                if runtime.is_none() {
                    match spawn_python_tool_server_runtime(worker_id, parent_pid).await {
                        Ok(new_runtime) => {
                            runtime = Some(new_runtime);
                        }
                        Err(error) => {
                            let _ =
                                request
                                    .response_tx
                                    .send(PythonToolResponse::PythonError(format!(
                                        "Failed to spawn python tool server worker {}: {}",
                                        worker_id, error
                                    )));
                            continue;
                        }
                    }
                }

                let worker_runtime = runtime
                    .as_mut()
                    .expect("runtime should exist after spawn success");
                let (response, reset_runtime) = execute_request_on_runtime(
                    worker_runtime,
                    next_request_id,
                    request.code,
                    worker_id,
                )
                .await;
                next_request_id = next_request_id.saturating_add(1);

                if reset_runtime {
                    if let Some(mut existing_runtime) = runtime.take() {
                        shutdown_runtime(&mut existing_runtime).await;
                    }
                }
                let _ = request.response_tx.send(response);
            }

            if let Some(mut existing_runtime) = runtime.take() {
                shutdown_runtime(&mut existing_runtime).await;
            }
        });

        Ok(Self {
            request_tx,
            inflight: AtomicUsize::new(0),
        })
    }

    async fn execute(&self, code: String) -> PythonToolResponse {
        self.inflight.fetch_add(1, Ordering::SeqCst);
        let (response_tx, response_rx) = oneshot::channel();
        let queued = self
            .request_tx
            .send(ExecuteRequest { code, response_tx })
            .await;
        if queued.is_err() {
            self.inflight.fetch_sub(1, Ordering::SeqCst);
            return PythonToolResponse::PythonError(
                "Python tool worker queue closed unexpectedly.".to_string(),
            );
        }
        let response = match response_rx.await {
            Ok(response) => response,
            Err(_) => PythonToolResponse::PythonError(
                "Python tool worker response channel closed unexpectedly.".to_string(),
            ),
        };
        self.inflight.fetch_sub(1, Ordering::SeqCst);
        response
    }

    fn inflight(&self) -> usize {
        self.inflight.load(Ordering::SeqCst)
    }
}

pub struct PythonToolServerPool {
    workers: Vec<Arc<PythonToolWorkerHandle>>,
    round_robin_cursor: AtomicUsize,
}

impl PythonToolServerPool {
    pub async fn new(num_servers: usize) -> Result<Self, String> {
        if num_servers == 0 {
            return Err("num_servers must be greater than zero".to_string());
        }
        let parent_pid = std::process::id();
        let mut workers = Vec::with_capacity(num_servers);
        for worker_id in 0..num_servers {
            let worker = PythonToolWorkerHandle::spawn(worker_id, parent_pid).await?;
            workers.push(Arc::new(worker));
        }
        Ok(Self {
            workers,
            round_robin_cursor: AtomicUsize::new(0),
        })
    }

    fn pick_worker_index(&self) -> usize {
        let num_workers = self.workers.len();
        let start = self.round_robin_cursor.fetch_add(1, Ordering::Relaxed) % num_workers;
        let mut best_index = start;
        let mut best_inflight = usize::MAX;
        for offset in 0..num_workers {
            let index = (start + offset) % num_workers;
            let inflight = self.workers[index].inflight();
            if inflight < best_inflight {
                best_inflight = inflight;
                best_index = index;
            }
        }
        best_index
    }

    pub async fn execute_code(&self, code: String) -> PythonToolResponse {
        let worker_index = self.pick_worker_index();
        let raw_response = self.workers[worker_index].execute(code).await;
        match raw_response {
            PythonToolResponse::PythonSuccess(output) => {
                if output.trim().is_empty() {
                    log_warning("Python interpreter did not return any output.");
                    return PythonToolResponse::PythonSuccess(
                        "Python interpreter did not return any output. Please use print statements to retrieve results.".to_string(),
                    );
                }
                PythonToolResponse::PythonSuccess(output)
            }
            PythonToolResponse::PythonError(error) => PythonToolResponse::PythonError(error),
        }
    }
}

async fn spawn_python_tool_server_runtime(
    worker_id: usize,
    parent_pid: u32,
) -> Result<PythonToolWorkerRuntime, String> {
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("-m")
        .arg("src_py.tool_server.main")
        .arg("--parent-pid")
        .arg(parent_pid.to_string())
        .arg("--worker-id")
        .arg(worker_id.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to spawn uv python tool server process: {}", error))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to capture python tool server stdin".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture python tool server stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture python tool server stderr".to_string())?;

    let stderr_task = tokio::spawn(stream_worker_stderr(stderr, worker_id));
    Ok(PythonToolWorkerRuntime {
        child,
        stdin,
        stdout_lines: BufReader::new(stdout).lines(),
        stderr_task,
    })
}

async fn stream_worker_stderr(stderr: ChildStderr, worker_id: usize) {
    let mut lines = BufReader::new(stderr).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                if line.trim().is_empty() {
                    continue;
                }
                log_warning(format!("[PY_TOOL_WORKER:{}] {}", worker_id, line));
            }
            Ok(None) => {
                break;
            }
            Err(error) => {
                log_warning(format!(
                    "[PY_TOOL_WORKER:{}] failed to read stderr: {}",
                    worker_id, error
                ));
                break;
            }
        }
    }
}

async fn shutdown_runtime(runtime: &mut PythonToolWorkerRuntime) {
    let _ = runtime.stdin.shutdown().await;
    let _ = timeout(Duration::from_millis(200), runtime.child.wait()).await;
    let _ = runtime.child.start_kill();
    let _ = runtime.child.wait().await;
    runtime.stderr_task.abort();
}

async fn execute_request_on_runtime(
    runtime: &mut PythonToolWorkerRuntime,
    request_id: u64,
    code: String,
    worker_id: usize,
) -> (PythonToolResponse, bool) {
    let request_wire = PythonToolRequestWire {
        id: request_id,
        code,
        timeout_ms: PYTHON_TOOL_TIMEOUT_MS,
    };
    let request_line = match serde_json::to_string(&request_wire) {
        Ok(serialized) => serialized,
        Err(error) => {
            return (
                PythonToolResponse::PythonError(format!(
                    "Failed to serialize python tool request: {}",
                    error
                )),
                false,
            );
        }
    };

    if let Err(error) = runtime.stdin.write_all(request_line.as_bytes()).await {
        return (
            PythonToolResponse::PythonError(format!(
                "Failed to write python tool request to worker {}: {}",
                worker_id, error
            )),
            true,
        );
    }
    if let Err(error) = runtime.stdin.write_all(b"\n").await {
        return (
            PythonToolResponse::PythonError(format!(
                "Failed to finalize python tool request to worker {}: {}",
                worker_id, error
            )),
            true,
        );
    }
    if let Err(error) = runtime.stdin.flush().await {
        return (
            PythonToolResponse::PythonError(format!(
                "Failed to flush python tool request to worker {}: {}",
                worker_id, error
            )),
            true,
        );
    }

    let response_line = runtime.stdout_lines.next_line().await;

    let line = match response_line {
        Ok(Some(line)) => line,
        Ok(None) => {
            return (
                PythonToolResponse::PythonError(format!(
                    "Python tool worker {} closed stdout unexpectedly.",
                    worker_id
                )),
                true,
            );
        }
        Err(error) => {
            return (
                PythonToolResponse::PythonError(format!(
                    "Failed to read python tool worker {} response: {}",
                    worker_id, error
                )),
                true,
            );
        }
    };

    let response_wire = match serde_json::from_str::<PythonToolResponseWire>(&line) {
        Ok(response_wire) => response_wire,
        Err(error) => {
            return (
                PythonToolResponse::PythonError(format!(
                    "Failed to parse python tool response from worker {}: {}",
                    worker_id, error
                )),
                true,
            );
        }
    };

    if response_wire.id != request_id {
        return (
            PythonToolResponse::PythonError(format!(
                "Python tool response id mismatch from worker {}: expected {}, got {}",
                worker_id, request_id, response_wire.id
            )),
            true,
        );
    }

    let response = if response_wire.ok {
        PythonToolResponse::PythonSuccess(response_wire.output.unwrap_or_default())
    } else {
        PythonToolResponse::PythonError(
            response_wire
                .error
                .unwrap_or_else(|| "Unknown python tool worker error".to_string()),
        )
    };
    (response, false)
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
