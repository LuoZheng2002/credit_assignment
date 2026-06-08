use std::{path::Path, time::Duration};

use research_utility::progress_tui_logger::log_info;
use serde::{Deserialize, Serialize};

use crate::{
    modal_training_client::{ModalTrainStartRequest, ModalTrainStartResponse},
    training_job_client::TrainingJobClient,
};

#[derive(Clone, Debug)]
pub struct HpcTrainingClient {
    client: reqwest::Client,
    base_url: String,
    bearer_token: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HpcTrainStatusResponse {
    pub status: String,
    pub progress_message: Option<String>,
    pub progress_fraction: Option<f32>,
}

impl HpcTrainingClient {
    pub fn new(
        client: reqwest::Client,
        base_url: &str,
        auth_token_env_var: Option<&str>,
    ) -> Result<Self, String> {
        let base_url = base_url.trim().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err("HPC training base URL cannot be empty".to_string());
        }

        let bearer_token = if let Some(env_var) = auth_token_env_var {
            let env_var = env_var.trim();
            if env_var.is_empty() {
                return Err("HPC auth token env var name cannot be empty".to_string());
            }
            let token = std::env::var(env_var)
                .map_err(|_| format!("HPC auth token env var is not set: {}", env_var))?;
            let token = token.trim().to_string();
            if token.is_empty() {
                return Err(format!("HPC auth token env var is empty: {}", env_var));
            }
            Some(token)
        } else {
            None
        };

        Ok(Self {
            client,
            base_url,
            bearer_token,
        })
    }

    pub async fn start_training(
        &self,
        request: &ModalTrainStartRequest,
    ) -> Result<ModalTrainStartResponse, String> {
        self.send_json(
            reqwest::Method::POST,
            &format!("{}/train/start", self.base_url),
            Some(request),
        )
        .await
    }

    pub async fn upload_trajectory_file(&self, job_id: &str, file_path: &str) -> Result<(), String> {
        let path = Path::new(file_path);
        if !path.is_file() {
            return Err(format!(
                "Training trajectory file does not exist for HPC upload: {}",
                file_path
            ));
        }
        let bytes = tokio::fs::read(path).await.map_err(|err| {
            format!(
                "Failed to read training trajectory file for HPC upload ({}): {}",
                file_path, err
            )
        })?;

        let url = format!("{}/train/upload_trajectory/{}", self.base_url, job_id.trim());
        let mut request = self
            .client
            .put(url.clone())
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .body(bytes);
        if let Some(token) = &self.bearer_token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|err| format!("HPC request failed to {}: {}", url, err))?;
        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(format!(
                "HPC upload failed (status {}) for {}: {}",
                status, url, body_text
            ));
        }
        Ok(())
    }

    pub async fn wait_until_done(
        &self,
        job_id: &str,
        poll_interval: Duration,
    ) -> Result<(), String> {
        let mut last_status = String::new();
        let mut last_message = String::new();
        loop {
            let status = self.get_status(job_id).await?;
            let status_changed = status.status != last_status;
            let message = status.progress_message.clone().unwrap_or_default();
            let message_changed = !message.is_empty() && message != last_message;

            if status_changed || message_changed {
                if let Some(fraction) = status.progress_fraction {
                    log_info(format!(
                        "HPC training status: {} ({:.2}%) {}",
                        status.status,
                        fraction * 100.0,
                        message
                    ));
                } else {
                    log_info(format!("HPC training status: {} {}", status.status, message));
                }
                last_status = status.status.clone();
                if !message.is_empty() {
                    last_message = message;
                }
            }

            match status.status.as_str() {
                "queued" | "starting" | "running" => {}
                "succeeded" => return Ok(()),
                "failed" | "cancelled" => {
                    return Err(format!(
                        "HPC training job {} finished with status {}",
                        job_id, status.status
                    ));
                }
                other => {
                    return Err(format!(
                        "HPC training job {} returned unknown status '{}'",
                        job_id, other
                    ));
                }
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    async fn get_status(&self, job_id: &str) -> Result<HpcTrainStatusResponse, String> {
        self.send_json::<(), HpcTrainStatusResponse>(
            reqwest::Method::GET,
            &format!("{}/train/status/{}", self.base_url, job_id.trim()),
            None,
        )
        .await
    }

    async fn send_json<TRequest, TResponse>(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<&TRequest>,
    ) -> Result<TResponse, String>
    where
        TRequest: Serialize,
        TResponse: for<'de> Deserialize<'de>,
    {
        let mut request = self.client.request(method, url);
        if let Some(token) = &self.bearer_token {
            request = request.bearer_auth(token);
        }
        if let Some(payload) = body {
            request = request.json(payload);
        }
        let response = request
            .send()
            .await
            .map_err(|err| format!("HPC request failed to {}: {}", url, err))?;
        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(format!(
                "HPC request failed (status {}) for {}: {}",
                status, url, body_text
            ));
        }
        response
            .json::<TResponse>()
            .await
            .map_err(|err| format!("HPC response JSON parse failed for {}: {}", url, err))
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn hpc_training_client_contract_smoke_test() {
        let mock_server = MockServer::start().await;

        let start_response = serde_json::json!({"job_id": "hpc-job-123"});
        Mock::given(method("POST"))
            .and(path("/train/start"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(start_response))
            .mount(&mock_server)
            .await;

        Mock::given(method("PUT"))
            .and(path("/train/upload_trajectory/hpc-job-123"))
            .and(header("content-type", "application/octet-stream"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let status_response = serde_json::json!({"status": "succeeded"});
        Mock::given(method("GET"))
            .and(path("/train/status/hpc-job-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(status_response))
            .mount(&mock_server)
            .await;

        let client = HpcTrainingClient::new(reqwest::Client::new(), &mock_server.uri(), None)
            .expect("construct hpc client");

        let request = ModalTrainStartRequest {
            model_cli_name: "qwen35_4b".to_string(),
            config_nickname: "cfg".to_string(),
            epoch: 1,
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
                artifact_root_dir: "/tmp/storage".to_string(),
                hpc_training_root_dir: Some("/tmp/hpc_training_root".to_string()),
                model_cli_name: "qwen35_4b".to_string(),
                config_nickname: "cfg".to_string(),
                epoch: 1,
            },
        };

        let started = client
            .start_training(&request)
            .await
            .expect("start training request should succeed");
        assert_eq!(started.job_id, "hpc-job-123");

        let temp_sqlite = NamedTempFile::new().expect("create temp sqlite file");
        tokio::fs::write(temp_sqlite.path(), b"sqlite-bytes")
            .await
            .expect("write sqlite payload");

        client
            .upload_trajectory_file(&started.job_id, temp_sqlite.path().to_str().expect("utf8 path"))
            .await
            .expect("upload should succeed");

        client
            .wait_until_done(&started.job_id, Duration::from_millis(10))
            .await
            .expect("status polling should complete");
    }
}
