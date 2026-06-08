use std::{path::Path, time::Duration};

use rand::RngExt;
use research_utility::progress_tui_logger::log_info;
use serde::{Deserialize, Serialize};

use crate::{
    json_line_util::write_toml,
    launch_python_training::launch_python_training_process,
    modal_training_client::{ModalTrainStartRequest, ModalTrainStartResponse},
    training_job_client::TrainingJobClient,
};

#[derive(Clone, Debug)]
pub struct HpcTrainingClient {
    job_root_dir: String,
    num_gpus: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HpcTrainingJobStatus {
    pub status: String,
    pub message: String,
}

impl HpcTrainingClient {
    pub fn new(job_root_dir: &str, num_gpus: usize) -> Result<Self, String> {
        assert!(num_gpus > 0, "num_gpus must be positive");
        let root = job_root_dir.trim();
        if root.is_empty() {
            return Err("HPC training job root directory cannot be empty".to_string());
        }
        std::fs::create_dir_all(root)
            .map_err(|err| format!("Failed to create HPC training job root {}: {}", root, err))?;
        Ok(Self {
            job_root_dir: root.to_string(),
            num_gpus,
        })
    }

    pub async fn start_training(
        &self,
        request: &ModalTrainStartRequest,
    ) -> Result<ModalTrainStartResponse, String> {
        let job_id = self.generate_job_id(
            &request.model_cli_name,
            &request.config_nickname,
            request.epoch,
        );
        let job_folder = self.job_folder_path(&job_id);
        std::fs::create_dir_all(format!("{}/input", job_folder)).map_err(|err| {
            format!(
                "Failed to create HPC training job input folder {}: {}",
                job_folder, err
            )
        })?;
        let request_path = format!("{}/train_request.toml", job_folder);
        write_toml(&request_path, &request.training_config)?;
        self.write_status(
            &job_id,
            "queued",
            "Job created and waiting for local execution",
        )?;
        log_info(format!(
            "Created HPC training job {} at {}",
            job_id, job_folder
        ));
        Ok(ModalTrainStartResponse { job_id })
    }

    pub async fn upload_trajectory_file(&self, job_id: &str, file_path: &str) -> Result<(), String> {
        let source = Path::new(file_path);
        if !source.is_file() {
            return Err(format!(
                "Training trajectory file does not exist for HPC upload: {}",
                file_path
            ));
        }
        let destination = format!(
            "{}/input/training_trajectories.sqlite",
            self.job_folder_path(job_id)
        );
        tokio::fs::copy(source, &destination).await.map_err(|err| {
            format!(
                "Failed to copy trajectory sqlite for HPC job {} ({} -> {}): {}",
                job_id, file_path, destination, err
            )
        })?;
        Ok(())
    }

    pub async fn wait_until_done(
        &self,
        job_id: &str,
        poll_interval: Duration,
    ) -> Result<(), String> {
        let job_folder = self.job_folder_path(job_id);
        log_info(format!(
            "Starting HPC training job {} with isolated folder {}",
            job_id, job_folder
        ));
        self.write_status(job_id, "running", "Launching local torchrun job")?;

        let test_mode_path = format!("{}/.test_mode", job_folder);
        if Path::new(&test_mode_path).is_file() {
            self.write_status(job_id, "succeeded", "Test mode completed successfully")?;
            return Ok(());
        }

        let mut handle = launch_python_training_process(self.num_gpus, job_folder).await;
        let exit_status = loop {
            self.request_log_flush(job_id)?;
            match handle
                .process
                .try_wait()
                .map_err(|err| format!("Failed to poll python training process status: {}", err))?
            {
                Some(status) => break status,
                None => tokio::time::sleep(poll_interval).await,
            }
        };
        self.request_log_flush(job_id)?;
        handle
            .io_listener_task
            .await
            .map_err(|err| format!("Python training log listener task failed: {}", err))?;
        if !exit_status.success() {
            self.write_status(
                job_id,
                "failed",
                &format!("python training exited with status {}", exit_status),
            )?;
            return Err(format!(
                "Python training process exited with non-zero status: {}",
                exit_status
            ));
        }
        self.write_status(job_id, "succeeded", "Local training process completed successfully")?;
        Ok(())
    }

    fn generate_job_id(&self, model_cli_name: &str, config_nickname: &str, epoch: usize) -> String {
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let mut rng = rand::rng();
        let suffix: u32 = rng.random();
        format!(
            "hpc-{}-{}-epoch{}-{}-{:08x}",
            sanitize_segment(model_cli_name),
            sanitize_segment(config_nickname),
            epoch,
            millis,
            suffix
        )
    }

    fn job_folder_path(&self, job_id: &str) -> String {
        format!("{}/{}", self.job_root_dir, job_id.trim())
    }

    fn status_file_path(&self, job_id: &str) -> String {
        format!("{}/status.json", self.job_folder_path(job_id))
    }

    fn log_flush_request_file_path(&self, job_id: &str) -> String {
        format!("{}/.flush_logs.request", self.job_folder_path(job_id))
    }

    fn write_status(&self, job_id: &str, status: &str, message: &str) -> Result<(), String> {
        let payload = HpcTrainingJobStatus {
            status: status.to_string(),
            message: message.to_string(),
        };
        let serialized = serde_json::to_string_pretty(&payload)
            .map_err(|err| format!("Failed to serialize HPC job status: {}", err))?;
        std::fs::write(self.status_file_path(job_id), serialized)
            .map_err(|err| format!("Failed to write HPC job status for {}: {}", job_id, err))
    }

    fn request_log_flush(&self, job_id: &str) -> Result<(), String> {
        std::fs::write(self.log_flush_request_file_path(job_id), b"flush")
            .map_err(|err| format!("Failed to request log flush for HPC job {}: {}", job_id, err))
    }
}

#[async_trait::async_trait]
impl TrainingJobClient for HpcTrainingClient {
    fn backend_label(&self) -> &'static str {
        "HPC"
    }

    async fn start_training(
        &self,
        request: &ModalTrainStartRequest,
    ) -> Result<ModalTrainStartResponse, String> {
        HpcTrainingClient::start_training(self, request).await
    }

    async fn upload_trajectory_file(&self, job_id: &str, file_path: &str) -> Result<(), String> {
        HpcTrainingClient::upload_trajectory_file(self, job_id, file_path).await
    }

    async fn wait_until_done(&self, job_id: &str, poll_interval: Duration) -> Result<(), String> {
        HpcTrainingClient::wait_until_done(self, job_id, poll_interval).await
    }
}

fn sanitize_segment(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn hpc_training_client_smoke_test() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let client = HpcTrainingClient::new(temp_dir.path().to_str().expect("temp dir utf8"), 1)
            .expect("build hpc client");

        let request = ModalTrainStartRequest {
            model_cli_name: "qwen35_4b".to_string(),
            config_nickname: "test_run".to_string(),
            epoch: 2,
            num_gpus: 1,
            training_config: crate::python_training_config::PythonTrainingConfig {
                common: crate::python_training_config::PythonTrainingConfigCommon {
                    training_plan: "lora".to_string(),
                    advantage_clip: 3.0,
                    learning_rate: 1e-5,
                    weight_decay: 0.0,
                    grad_accum_steps: 1,
                    log_time_interval: 1.0,
                    checkpoint_save_time_interval: 60.0,
                    seed: 42,
                    lora_rank: Some(64),
                    lora_alpha: Some(128),
                    lora_dropout: Some(0.05),
                    lora_target_modules_csv: Some("q_proj,k_proj,v_proj,o_proj".to_string()),
                    resume_checkpoint_tag: Some("auto".to_string()),
                },
                training_time: 10.0,
                num_iterations_limit: 100,
                storage_root_dir: "/tmp/storage".to_string(),
                model_cli_name: "qwen35_4b".to_string(),
                config_nickname: "test_run".to_string(),
                epoch: 2,
            },
        };
        let started = client
            .start_training(&request)
            .await
            .expect("start training job");

        let trajectory_src = temp_dir.path().join("trajectory.sqlite");
        std::fs::write(&trajectory_src, b"sqlite").expect("write trajectory src");
        client
            .upload_trajectory_file(&started.job_id, trajectory_src.to_str().expect("utf8 path"))
            .await
            .expect("upload trajectory");

        let test_mode_marker = temp_dir.path().join(&started.job_id).join(".test_mode");
        std::fs::write(&test_mode_marker, b"1").expect("create test mode marker");
        client
            .wait_until_done(&started.job_id, Duration::from_millis(10))
            .await
            .expect("wait should succeed in test mode");

        let status_path = temp_dir
            .path()
            .join(started.job_id)
            .join("status.json");
        let status_raw = std::fs::read_to_string(&status_path).expect("read status file");
        let status: HpcTrainingJobStatus =
            serde_json::from_str(&status_raw).expect("parse status json");
        assert_eq!(status.status, "succeeded");
    }
}
