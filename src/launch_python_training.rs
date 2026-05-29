use std::{
    io::{BufRead, BufReader, Read},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use research_utility::log_message::{log_info, log_warning};
use tokio::task::JoinHandle;

pub struct PythonTrainingProcessHandle {
    pub process: Child,
    pub io_listener_task: JoinHandle<()>,
    pub listener_should_stop: Arc<AtomicBool>,
}

pub async fn launch_python_training_process(
    num_gpus: usize,
    training_config_path: String,
) -> PythonTrainingProcessHandle {
    assert!(num_gpus > 0, "num_gpus must be positive");
    let config_path = training_config_path.trim();
    assert!(
        !config_path.is_empty(),
        "training_config_path cannot be empty"
    );
    assert!(
        Path::new(config_path).is_file(),
        "training config file does not exist: {}",
        config_path
    );

    let master_port = std::env::var("MASTER_PORT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "29501".to_string());

    log_info(format!(
        "Launching python training with torchrun (num_gpus={}, master_port={})",
        num_gpus, master_port
    ));

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("--project")
        .arg("pyprojects/common")
        .arg("torchrun")
        .arg("--nproc_per_node")
        .arg(num_gpus.to_string())
        .arg("--master_port")
        .arg(master_port)
        .arg("-m")
        .arg("src_py.train.main_from_config")
        .arg("--config-toml-path")
        .arg(config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut process = command.spawn().unwrap_or_else(|err| {
        panic!(
            "Failed to launch python training process with config {}: {}",
            config_path, err
        )
    });

    let stdout = process.stdout.take().unwrap_or_else(|| {
        panic!(
            "Failed to capture stdout for python training process ({})",
            config_path
        )
    });
    let stderr = process.stderr.take().unwrap_or_else(|| {
        panic!(
            "Failed to capture stderr for python training process ({})",
            config_path
        )
    });

    let listener_should_stop = Arc::new(AtomicBool::new(false));
    let stdout_listener = spawn_pipe_listener(stdout, listener_should_stop.clone(), false);
    let stderr_listener = spawn_pipe_listener(stderr, listener_should_stop.clone(), true);

    let io_listener_task = tokio::spawn(async move {
        let _ = stdout_listener.await;
        let _ = stderr_listener.await;
    });

    PythonTrainingProcessHandle {
        process,
        io_listener_task,
        listener_should_stop,
    }
}

fn spawn_pipe_listener<R: Read + Send + 'static>(
    reader: R,
    listener_should_stop: Arc<AtomicBool>,
    is_stderr: bool,
) -> JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let mut buffered = BufReader::new(reader);
        loop {
            if listener_should_stop.load(Ordering::Relaxed) {
                break;
            }

            let mut line = String::new();
            match buffered.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let content = line.trim_end_matches(['\n', '\r']);
                    if content.is_empty() {
                        continue;
                    }
                    if is_stderr {
                        log_warning(format!("[TRAIN]: {}", content));
                    } else {
                        log_info(format!("[TRAIN]: {}", content));
                    }
                }
                Err(err) => {
                    log_warning(format!("[TRAIN]: Failed to read process output: {}", err));
                    break;
                }
            }
        }
    })
}
