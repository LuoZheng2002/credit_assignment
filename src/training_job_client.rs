use std::time::Duration;

use async_trait::async_trait;

use crate::modal_training_client::{ModalTrainStartRequest, ModalTrainStartResponse};

#[async_trait]
pub trait TrainingJobClient: Send + Sync {
    fn backend_label(&self) -> &'static str;

    async fn start_training(
        &self,
        request: &ModalTrainStartRequest,
    ) -> Result<ModalTrainStartResponse, String>;

    async fn upload_trajectory_file(&self, job_id: &str, file_path: &str) -> Result<(), String>;

    async fn wait_until_done(&self, job_id: &str, poll_interval: Duration) -> Result<(), String>;
}
