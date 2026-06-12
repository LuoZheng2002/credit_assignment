use std::{process::Stdio, time::Duration};

use research_utility::progress_tui_logger::log_warning;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
    sync::Semaphore,
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

const PYTHON_TOOL_TIMEOUT_MS: u64 = 5000;
const PYTHON_TOOL_PROCESS_SHUTDOWN_TIMEOUT_MS: u64 = 1000;
const PYTHON_TOOL_PIPE_DRAIN_TIMEOUT_MS: u64 = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonToolResponseWire {
    ok: bool,
    output: Option<String>,
    error: Option<String>,
}

pub struct PythonToolServerPool {
    process_limiter: std::sync::Arc<Semaphore>,
}

impl PythonToolServerPool {
    pub async fn new(max_python_processes: usize) -> Result<Self, String> {
        if max_python_processes == 0 {
            return Err("max_python_processes must be greater than zero".to_string());
        }
        Ok(Self {
            process_limiter: std::sync::Arc::new(Semaphore::new(max_python_processes)),
        })
    }

    pub async fn execute_code(&self, code: String) -> PythonToolResponse {
        let permit = match self.process_limiter.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                return PythonToolResponse::PythonError(
                    "Python process limiter closed unexpectedly.".to_string(),
                );
            }
        };
        let raw_response = execute_code_in_fresh_interpreter(code).await;
        drop(permit);

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

async fn read_pipe_to_string<T>(pipe: T) -> String
where
    T: AsyncRead + Unpin,
{
    let mut reader = pipe;
    let mut buffer = Vec::new();
    if reader.read_to_end(&mut buffer).await.is_err() {
        return String::new();
    }
    String::from_utf8_lossy(&buffer).to_string()
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

async fn await_pipe_task(task: JoinHandle<String>, label: &str) -> Result<String, String> {
    match timeout(
        Duration::from_millis(PYTHON_TOOL_PIPE_DRAIN_TIMEOUT_MS),
        task,
    )
    .await
    {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(join_error)) => Err(format!(
            "Failed to join {} reader task: {}",
            label, join_error
        )),
        Err(_) => {
            log_warning(format!(
                "Timed out draining python interpreter {} within {} ms after process exit; a descendant process may still be holding the pipe open.",
                label, PYTHON_TOOL_PIPE_DRAIN_TIMEOUT_MS
            ));
            Err(format!(
                "Timed out while draining python interpreter {} after process exit; a descendant process may still be holding the pipe open.",
                label
            ))
        }
    }
}

async fn execute_code_in_fresh_interpreter(code: String) -> PythonToolResponse {
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("-m")
        .arg("src_py.tool_server.main")
        .arg("--single-shot")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(unix)]
    command.process_group(0);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return PythonToolResponse::PythonError(format!(
                "Failed to spawn python interpreter process: {}",
                error
            ));
        }
    };

    let child_pid = child.id();

    let mut stdin = match child.stdin.take() {
        Some(stdin) => stdin,
        None => {
            terminate_tool_process(&mut child, child_pid).await;
            return PythonToolResponse::PythonError(
                "Failed to capture python interpreter stdin".to_string(),
            );
        }
    };
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            terminate_tool_process(&mut child, child_pid).await;
            return PythonToolResponse::PythonError(
                "Failed to capture python interpreter stdout".to_string(),
            );
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            terminate_tool_process(&mut child, child_pid).await;
            return PythonToolResponse::PythonError(
                "Failed to capture python interpreter stderr".to_string(),
            );
        }
    };

    let stdout_task = tokio::spawn(read_pipe_to_string(stdout));
    let stderr_task = tokio::spawn(read_pipe_to_string(stderr));

    if let Err(error) = stdin.write_all(code.as_bytes()).await {
        terminate_tool_process(&mut child, child_pid).await;
        return PythonToolResponse::PythonError(format!(
            "Failed to write python code to interpreter stdin: {}",
            error
        ));
    }
    if let Err(error) = stdin.shutdown().await {
        terminate_tool_process(&mut child, child_pid).await;
        return PythonToolResponse::PythonError(format!(
            "Failed to close python interpreter stdin: {}",
            error
        ));
    }

    let wait_result = timeout(Duration::from_millis(PYTHON_TOOL_TIMEOUT_MS), child.wait()).await;
    match wait_result {
        Ok(Ok(_status)) => {
            let stdout_text = match await_pipe_task(stdout_task, "stdout").await {
                Ok(output) => output,
                Err(error) => {
                    return PythonToolResponse::PythonError(error);
                }
            };
            let stderr_text = match await_pipe_task(stderr_task, "stderr").await {
                Ok(output) => output,
                Err(error) => {
                    return PythonToolResponse::PythonError(error);
                }
            };
            parse_python_response(stdout_text, stderr_text)
        }
        Ok(Err(error)) => {
            terminate_tool_process(&mut child, child_pid).await;
            let stdout_result = await_pipe_task(stdout_task, "stdout").await;
            let stderr_result = await_pipe_task(stderr_task, "stderr").await;
            let mut message = format!(
                "Failed while waiting for python interpreter process: {}",
                error
            );
            if let Err(stdout_error) = stdout_result {
                message.push_str(&format!(" stdout_drain_error={}", stdout_error));
            }
            if let Err(stderr_error) = stderr_result {
                message.push_str(&format!(" stderr_drain_error={}", stderr_error));
            }
            PythonToolResponse::PythonError(message)
        }
        Err(_) => {
            terminate_tool_process(&mut child, child_pid).await;
            let stdout_result = await_pipe_task(stdout_task, "stdout").await;
            let stderr_result = await_pipe_task(stderr_task, "stderr").await;
            let mut message = format!(
                "Python code execution timed out after {} ms.",
                PYTHON_TOOL_TIMEOUT_MS
            );
            if let Err(stdout_error) = stdout_result {
                message.push_str(&format!(" stdout_drain_error={}", stdout_error));
            }
            if let Err(stderr_error) = stderr_result {
                message.push_str(&format!(" stderr_drain_error={}", stderr_error));
            }
            PythonToolResponse::PythonError(message)
        }
    }
}

fn parse_python_response(stdout_text: String, stderr_text: String) -> PythonToolResponse {
    let trimmed_stdout = stdout_text.trim();
    if trimmed_stdout.is_empty() {
        let stderr = stderr_text.trim();
        if stderr.is_empty() {
            return PythonToolResponse::PythonError(
                "Python interpreter returned an empty response.".to_string(),
            );
        }
        return PythonToolResponse::PythonError(format!(
            "Python interpreter returned no structured response. stderr: {}",
            stderr
        ));
    }

    match serde_json::from_str::<PythonToolResponseWire>(trimmed_stdout) {
        Ok(response_wire) => {
            if response_wire.ok {
                PythonToolResponse::PythonSuccess(response_wire.output.unwrap_or_default())
            } else {
                PythonToolResponse::PythonError(
                    response_wire.error.unwrap_or_else(|| {
                        "Unknown python interpreter execution error".to_string()
                    }),
                )
            }
        }
        Err(error) => {
            let stderr = stderr_text.trim();
            if stderr.is_empty() {
                PythonToolResponse::PythonError(format!(
                    "Failed to parse python interpreter response as JSON: {}. Raw stdout: {}",
                    error, trimmed_stdout
                ))
            } else {
                PythonToolResponse::PythonError(format!(
                    "Failed to parse python interpreter response as JSON: {}. Raw stdout: {}. stderr: {}",
                    error, trimmed_stdout, stderr
                ))
            }
        }
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
    use super::{PYTHON_TOOL_TIMEOUT_MS, PythonToolResponse, PythonToolServerPool};
    use tokio::time::{Duration, Instant, timeout};

    #[tokio::test]
    async fn python_tool_timeout_does_not_hang_when_descendant_inherits_stdio() {
        let pool = PythonToolServerPool::new(1).await.unwrap();
        let code = r#"
import subprocess
import sys

subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
while True:
    pass
"#;
        let started = Instant::now();
        let response = timeout(
            Duration::from_millis(PYTHON_TOOL_TIMEOUT_MS + 4000),
            pool.execute_code(code.to_string()),
        )
        .await
        .expect("tool execution should return promptly after timeout handling");
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(PYTHON_TOOL_TIMEOUT_MS + 4000),
            "tool execution took too long: {:?}",
            elapsed
        );
        match response {
            PythonToolResponse::PythonError(message) => {
                assert!(
                    message.contains("timed out")
                        || message.contains("draining python interpreter"),
                    "unexpected error message: {}",
                    message
                );
            }
            other => panic!("expected timeout error, got {:?}", other),
        }
    }
}
