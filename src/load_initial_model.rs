use std::process::Stdio;

use research_utility::progress_tui_server::{log_info, log_warning};
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;

pub async fn load_initial_model(parent_dir: &str, hf_model_name: &str) -> Result<(), String> {
    log_info(format!(
        "Loading initial model {} to parent dir {}",
        hf_model_name, parent_dir,
    ));
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("--project")
        .arg("pyprojects/common")
        .arg("scripts/load_model_to_path.py")
        .arg("--output-parent-dir")
        .arg(parent_dir)
        .arg("--model")
        .arg(hf_model_name)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|err| format!("Failed to spawn model loader process: {}", err))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout_task = tokio::spawn(stream_output(stdout, "[LOAD_MODEL]"));
    let stderr_task = tokio::spawn(stream_output(stderr, "[LOAD_MODEL]"));
    let io_task = tokio::spawn(async move {
        let _ = stdout_task.await;
        let _ = stderr_task.await;
    });
    let exit_status = child
        .wait()
        .await
        .map_err(|err| format!("Failed to wait for model loader process: {}", err))?;
    io_task
        .await
        .map_err(|err| format!("Model loader log listener task failed: {}", err))?;
    if !exit_status.success() {
        return Err(format!(
            "Model loader process exited with non-zero status: {}",
            exit_status
        ));
    }
    log_info("Initial model loaded successfully");
    Ok(())
}

async fn stream_output<R>(reader: R, prefix: &'static str)
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut lines = tokio::io::BufReader::new(reader).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(content)) => {
                if content.is_empty() {
                    continue;
                }
                log_info(format!("{}: {}", prefix, content));
            }
            Ok(None) => break,
            Err(err) => {
                log_warning(format!("{} Failed to read process output: {}", prefix, err));
                break;
            }
        }
    }
}
