use std::{process::Child, sync::{Arc, atomic::Ordering}};

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
    json_line_util::{write_json, write_toml},
    launch_python_training::launch_python_training_process,
    launch_sglang_server::{
        launch_sglang_server_process, model_uses_sglang, shut_down_sglang_server_process,
    },
    llm_model::{LlmCliArgs, LlmModelMarker},
    python_training_config::{PythonTrainingConfig, PythonTrainingConfigCommon},
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
    // for training
    pub training_config_common: PythonTrainingConfigCommon,
    pub first_n_training_samples: Option<usize>,
    pub num_gpus: usize,
}

pub struct InferenceServerHandle {
    pub epoch: usize,
    pub sglang_port: Option<u16>,
    pub process: Option<Child>,
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
                    self.ensure_inference_server_launched::<M>(epoch).await;
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
                    self.ensure_inference_server_launched::<M>(epoch).await;
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
                    self.train_model::<M>(epoch)
                        .await
                        .expect("Python training failed");
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

    async fn ensure_inference_server_launched<M: LlmModelMarker>(&mut self, epoch: usize) {
        if let Some(handle) = &self.inference_server_handle {
            if handle.epoch == epoch {
                // already launched for this epoch
                return;
            } else {
                // first shut down the previous one
                self.ensure_inference_server_shut_down();
                // then continue to launch the new one
                self.launch_inference_server::<M>(epoch).await;
            }
        } else {
            // not launched, just launch
            self.launch_inference_server::<M>(epoch).await;
        }
    }

    async fn launch_inference_server<M: LlmModelMarker>(&mut self, epoch: usize) {
        assert!(
            self.inference_server_handle.is_none(),
            "Inference server is already launched for epoch {}, cannot launch again without shutting down",
            self.inference_server_handle.as_ref().unwrap().epoch
        );
        if !model_uses_sglang::<M>() {
            self.inference_server_handle = Some(InferenceServerHandle {
                epoch,
                sglang_port: None,
                process: None,
            });
            log_info(format!(
                "Model {} does not need a local inference server",
                M::CLI_NAME
            ));
            return;
        }

        log_info(format!(
            "Launching inference server for model {}",
            M::CLI_NAME,
        ));
        let model_path = self.model_folder_path::<M>(epoch);
        log_info(format!("Using model folder path: {}", model_path));
        let (sglang_port, process) =
            launch_sglang_server_process::<M>(&model_path, self.sglang_server_log_path.as_deref())
                .await;
        log_info(format!(
            "SGLang server is listening on port {}",
            sglang_port
        ));

        self.inference_server_handle = Some(InferenceServerHandle {
            epoch,
            sglang_port: Some(sglang_port),
            process: Some(process),
        });
        log_info("Inference server launched");
    }

    fn ensure_inference_server_shut_down(&mut self) {
        if let Some(handle) = self.inference_server_handle.take() {
            log_info("Shutting down inference server...");
            if let Some(mut process) = handle.process {
                shut_down_sglang_server_process(&mut process);
            }
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
    async fn train_model<M: LlmModelMarker>(&self, epoch: usize) -> Result<(), String> {
        let asset_file_training_trajectories = AssetFileTrainingTrajectories {
            config_nickname: self.config_nickname.clone(),
            epoch,
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            max_num_training_trajectories: self.max_num_training_trajectories,
            _phantom: std::marker::PhantomData::<M>,
        };
        let training_trajectory_sqlite_path = asset_file_training_trajectories.file_path();
        // first we need to write the training config to the expected location
        let training_config = PythonTrainingConfig {
            common: self.training_config_common.clone(),
            model_parent_dir: self.epoch_dir::<M>(epoch),
            training_trajectory_sqlite_path,
            checkpoints_parent_dir: self.epoch_dir::<M>(epoch),
            final_model_output_parent_dir: self.epoch_dir::<M>(epoch),
            first_n_training_samples: self.first_n_training_samples,
        };
        let training_config_path = self.python_training_config_path::<M>(epoch);
        write_toml(&training_config_path, &training_config).unwrap();
        // launch the training python code
        let mut handle = launch_python_training_process(self.num_gpus, training_config_path).await;
        // wait for it to finish, do not interrupt unless the process crashes
        let exit_status = handle
            .process
            .wait()
            .map_err(|err| format!("Failed to wait for python training process: {}", err))?;
        handle.listener_should_stop.store(true, Ordering::Relaxed);
        handle
            .io_listener_task
            .await
            .map_err(|err| format!("Python training log listener task failed: {}", err))?;
        if !exit_status.success() {
            return Err(format!(
                "Python training process exited with non-zero status: {}",
                exit_status
            ));
        }
        Ok(())
    }

    fn model_folder_path<M: LlmModelMarker>(&self, epoch: usize) -> String {
        format!(
            "results/{}/{}/epoch_{}/model",
            M::CLI_NAME,
            self.config_nickname,
            epoch
        )
    }
    fn epoch_dir<M: LlmModelMarker>(&self, epoch: usize) -> String {
        format!(
            "results/{}/{}/epoch_{}",
            M::CLI_NAME,
            self.config_nickname,
            epoch
        )
    }

    fn python_training_config_path<M: LlmModelMarker>(&self, epoch: usize) -> String {
        format!(
            "results/{}/{}/epoch_{}/python_training_config.json",
            M::CLI_NAME,
            self.config_nickname,
            epoch
        )
    }
}
impl Drop for Orchestrator {
    fn drop(&mut self) {
        self.ensure_inference_server_shut_down();
        log_info("Orchestrator dropped, inference server (if any) should be shut down");
        println!("Orchestrator dropped, inference server (if any) should be shut down");
    }
}
