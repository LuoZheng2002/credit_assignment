use std::path::Path;

use research_utility::progress_text_logger::log_info;

use serde::Serialize;

use research_utility::launch_python_process::PythonProcessLauncher;

use crate::python_training_config::PythonTrainingConfig;

#[derive(Serialize)]
struct TrainingWrapperStdinArgs {
    training_config: PythonTrainingConfig,
    num_gpus: usize,
    trajectory_path: String,
    hf_model_name: String,
    wrapper_log_path: String,
    #[serde(default)]
    test_sleep_secs: f32,
}

pub async fn run_training_wrapper_and_wait(
    num_gpus: usize,
    hf_model_name: &str,
    training_config: &PythonTrainingConfig,
    trajectory_path: &str,
    wrapper_log_path: &str,
) -> Result<(), String> {
    assert!(num_gpus > 0, "num_gpus must be positive");
    if !Path::new(trajectory_path).is_file() {
        return Err(format!(
            "training trajectory file does not exist: {}",
            trajectory_path
        ));
    }

    let args = TrainingWrapperStdinArgs {
        training_config: training_config.clone(),
        num_gpus,
        trajectory_path: trajectory_path.to_string(),
        hf_model_name: hf_model_name.to_string(),
        wrapper_log_path: wrapper_log_path.to_string(),
        test_sleep_secs: 0.0,
    };

    let handle = PythonProcessLauncher::new("training", "src_py.wrappers.training_wrapper")
        .with_stdin_json(&args)?
        .launch()
        .await?;

    let status = handle.wait_and_shutdown().await?;

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
