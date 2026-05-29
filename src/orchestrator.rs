use std::sync::Arc;

use research_utility::{asset_file::AssetFile, log_message::log_info};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::{
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::DirectRolloutConfig,
        direct_training_set::AssetFileTrainingTrajectories,
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::write_json,
    llm_model::{LlmCliArgs, LlmModelMarker},
};

pub struct Orchestrator {
    // for rollout
    pub config_nickname: String,
    pub validation_rollout_config: DirectRolloutConfig,
    pub training_set_rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub first_n_rollout_samples: Option<usize>,
    pub max_sqlite_connections: u32,
    pub inference_server_handle: Option<InferenceServerHandle>,
    pub sglang_server_log_path: Option<String>,
    // for training set generation
    pub max_num_training_trajectories: usize,
    // for orchestration
    pub num_total_epochs: usize,
    pub progress_save_file_path: String,
    // utilities
    pub client: reqwest::Client,
    pub question_semaphore: Arc<Semaphore>,
    // state
    pub progress: OrchestrationProgress,
    // for training: we assume only one set of training configuration, so no arguments
}

pub struct InferenceServerHandle {
    pub epoch: usize,
    pub sglang_port: Option<u16>,
    // to do: add process handle, etc.
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OrchestrationProgress {
    WorkingOnValidation { epoch: usize },
    WorkingOnRolloutCollection { epoch: usize },
    WorkingOnTrainingSetGeneration { epoch: usize },
    WorkingOnTraining { epoch: usize },
}

impl Orchestrator {
    pub async fn orchestrate<M: LlmModelMarker>(&mut self) {
        loop {
            match self.progress.clone() {
                OrchestrationProgress::WorkingOnValidation { epoch } => {
                    assert!(epoch <= self.num_total_epochs);
                    self.ensure_inference_server_launched(epoch);
                    self.validate_model::<M>(epoch).await;
                    if epoch >= self.num_total_epochs {
                        log_info(&format!(
                            "Finished all {} epochs of orchestration",
                            self.num_total_epochs
                        ));
                        self.ensure_inference_server_shut_down();
                        break;
                    }
                    self.update_and_save_progress(
                        OrchestrationProgress::WorkingOnRolloutCollection { epoch },
                    );
                }
                OrchestrationProgress::WorkingOnRolloutCollection { epoch } => {
                    self.ensure_inference_server_launched(epoch);
                    self.collect_training_rollout::<M>(epoch).await;
                    // after rollout collection, we can shut down the inference server
                    self.ensure_inference_server_shut_down();
                    self.update_and_save_progress(
                        OrchestrationProgress::WorkingOnTrainingSetGeneration { epoch },
                    );
                }
                OrchestrationProgress::WorkingOnTrainingSetGeneration { epoch } => {
                    // we do not need inference server for training set generation, and it won't be launched again until we do the training step
                    self.ensure_inference_server_shut_down();
                    self.generate_training_set::<M>(epoch).await;
                    self.update_and_save_progress(OrchestrationProgress::WorkingOnTraining {
                        epoch,
                    });
                }
                OrchestrationProgress::WorkingOnTraining { epoch } => {
                    // we do not want inference server to be up during training
                    self.ensure_inference_server_shut_down();
                    self.train_model::<M>(epoch).await;
                    assert!(epoch < self.num_total_epochs);
                    // do the final validation
                    self.update_and_save_progress(OrchestrationProgress::WorkingOnValidation {
                        epoch: epoch + 1,
                    });
                }
            }
            // for safety
            self.ensure_inference_server_shut_down();
        }
    }

    fn update_and_save_progress(&mut self, progress: OrchestrationProgress) {
        write_json(&self.progress_save_file_path, &progress).unwrap();
        self.progress = progress;
    }

    fn ensure_inference_server_launched(&mut self, epoch: usize) {
        if let Some(handle) = &self.inference_server_handle {
            if handle.epoch == epoch {
                // already launched for this epoch
                return;
            } else {
                // first shut down the previous one
                self.ensure_inference_server_shut_down();
                // then continue to launch the new one
                self.launch_inference_server(epoch);
            }
        } else {
            // not launched, just launch
            self.launch_inference_server(epoch);
        }
    }

    fn launch_inference_server(&mut self, epoch: usize) {
        assert!(
            self.inference_server_handle.is_none(),
            "Inference server is already launched for epoch {}, cannot launch again without shutting down",
            self.inference_server_handle.as_ref().unwrap().epoch
        );
        log_info("Launching inference server");
        // to do: spawn a server process like the one in scripts/sglang_serve/qwen35_08b.sh if the model requires sglang (local model)
        // for gpt model, we don't need to launch sglang server, and will panic at training stage
        // then fill the inference_server_handle, including the process handle

        // then redirect the stdout and stderr to the sglang_server_log_path if provided

        log_info("Inference server launched");
    }

    fn ensure_inference_server_shut_down(&mut self) {
        if let Some(handle) = self.inference_server_handle.take() {
            log_info("Shutting down inference server...");
            // kill the process
            log_info("Inference server shut down");
        }
        assert!(
            self.inference_server_handle.is_none(),
            "Inference server should be shut down, but it's still Some"
        );
    }

    async fn validate_model<M: LlmModelMarker>(&self, epoch: usize) {
        let validation_rollout_program_config = RolloutProgramConfig {
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.validation_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            client: self.client.clone(),
            question_semaphore: self.question_semaphore.clone(),
            llm_cli_args: LlmCliArgs {
                sglang_port: self
                    .inference_server_handle
                    .as_ref()
                    .and_then(|handle| handle.sglang_port),
            },
            first_n_samples: None,
            max_sqlite_connections: self.max_sqlite_connections,
        };
        rollout_all::<M>(validation_rollout_program_config).await;
    }

    async fn collect_training_rollout<M: LlmModelMarker>(&self, epoch: usize) {
        let Some(sglang_server_handle) = &self.inference_server_handle else {
            panic!("Orchestrator did not launch the sglang server before generating training set");
        };
        let llm_cli_args = LlmCliArgs {
            sglang_port: sglang_server_handle.sglang_port.clone(),
        };
        let training_set_rollout_program_config = RolloutProgramConfig {
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            client: self.client.clone(),
            question_semaphore: self.question_semaphore.clone(),
            llm_cli_args,
            first_n_samples: self.first_n_rollout_samples,
            max_sqlite_connections: self.max_sqlite_connections,
        };
        rollout_all::<M>(training_set_rollout_program_config).await;
    }

    async fn generate_training_set<M: LlmModelMarker>(&self, epoch: usize) {
        let asset_file_training_trajectories = AssetFileTrainingTrajectories {
            config_nickname: self.config_nickname.clone(),
            epoch,
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            max_num_training_trajectories: self.max_num_training_trajectories,
            _phantom: std::marker::PhantomData::<M>,
        };
        asset_file_training_trajectories.synchronize().await;
    }
    async fn train_model<M: LlmModelMarker>(&self, epoch: usize) {
        // launch the training python code
    }
}
impl Drop for Orchestrator {
    fn drop(&mut self) {
        self.ensure_inference_server_shut_down();
        log_info("Orchestrator dropped, inference server (if any) should be shut down");
        println!("Orchestrator dropped, inference server (if any) should be shut down");
    }
}
