use std::collections::BTreeMap;

use research_utility::{
    asset_file::AssetFile,
    log_message::{log_info, log_key_value_pair},
};
use serde::{Deserialize, Serialize};
use tokio::process::Child;

use crate::{
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
        direct_training_set::AssetFileTrainingTrajectories,
        direct_tree_action_log::AssetFileDirectTreeActionLogs,
        hybrid_dataset::DatasetSplit,
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    json_line_util::{write_json, write_toml},
    launch_python_training::launch_python_training_process,
    launch_sglang_server::{
        launch_sglang_server_process, model_uses_sglang, shut_down_sglang_server_process,
    },
    llm_model::{LlmCliArgs, LlmModelMarker},
    python_training_config::{PythonTrainingConfig, PythonTrainingConfigCommon},
    read_accuracy::read_accuracy,
};

pub struct Orchestrator {
    // for rollout
    pub config_nickname: String,
    pub validation_rollout_config: DirectRolloutConfig,
    pub training_set_rollout_config: DirectRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub training_rollout_time_limit_secs: usize,
    pub validation_rollout_time_limit_secs: usize,
    pub num_python_tool_servers: usize,
    pub inference_server_handle: Option<InferenceServerHandle>,
    pub sglang_server_log_path: Option<String>,
    // for training set generation
    pub cumulative_avg_abs_advantage_cutoff: f32,
    pub advantage_calculation_policy: AdvantageCalculationPolicy,
    // for orchestration
    pub num_total_epochs: usize,
    // utilities
    pub client: reqwest::Client,
    pub max_rollout_concurrency: usize,
    // state
    pub progress: OrchestrationProgress,
    // for training
    pub training_config_common: PythonTrainingConfigCommon,
    pub training_time: f32,
    pub num_iterations_limit: usize,
    pub num_gpus: usize,
}

pub struct InferenceServerHandle {
    pub epoch: usize,
    pub sglang_port: Option<u16>,
    pub process: Option<Child>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OrchestrationStatus {
    WorkingOnValidation,
    WorkingOnRolloutCollection,
    WorkingOnTrainingSetGeneration,
    WorkingOnTraining,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrchestrationProgress {
    pub status: OrchestrationStatus,
    pub epoch: usize,
    pub validation_accuracies: BTreeMap<usize, f32>,
}

impl Orchestrator {
    pub fn progress_save_path(model_cli_name: &str, config_nickname: &str) -> String {
        format!(
            "results/{}/{}/orchestration_progress.json",
            model_cli_name, config_nickname,
        )
    }
    pub async fn orchestrate<M: LlmModelMarker>(&mut self) -> Result<(), String> {
        loop {
            let progress = self.progress.clone();
            let epoch = progress.epoch;
            match progress.status {
                OrchestrationStatus::WorkingOnValidation => {
                    assert!(epoch <= self.num_total_epochs);
                    self.ensure_inference_server_launched::<M>(epoch).await?;
                    self.validate_model::<M>(epoch).await?;
                    self.read_and_log_validation_accuracy::<M>(epoch).await?;
                    if epoch >= self.num_total_epochs {
                        log_info(&format!(
                            "Finished all {} epochs of orchestration",
                            self.num_total_epochs
                        ));
                        self.ensure_inference_server_shut_down().await;
                        break;
                    }
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnRolloutCollection,
                        epoch,
                    );
                }
                OrchestrationStatus::WorkingOnRolloutCollection => {
                    self.ensure_inference_server_launched::<M>(epoch).await?;
                    self.collect_training_rollout::<M>(epoch).await;
                    // after rollout collection, we can shut down the inference server
                    self.ensure_inference_server_shut_down().await;
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnTrainingSetGeneration,
                        epoch,
                    );
                }
                OrchestrationStatus::WorkingOnTrainingSetGeneration => {
                    // we do not need inference server for training set generation, and it won't be launched again until we do the training step
                    self.ensure_inference_server_shut_down().await;
                    self.generate_training_set::<M>(epoch).await;
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnTraining,
                        epoch,
                    );
                }
                OrchestrationStatus::WorkingOnTraining => {
                    // we do not want inference server to be up during training
                    log_info(format!(
                        "Entering training stage for epoch {}. About to enforce inference-server shutdown.",
                        epoch
                    ));
                    self.ensure_inference_server_shut_down().await;
                    log_info(format!(
                        "Inference-server shutdown enforcement finished for epoch {}. Starting training.",
                        epoch
                    ));
                    self.train_model::<M>(epoch).await?;
                    assert!(epoch < self.num_total_epochs);
                    // do the final validation
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnValidation,
                        epoch + 1,
                    );
                }
            }
        }
        // for safety
        self.ensure_inference_server_shut_down().await;
        Ok(())
    }

    fn update_and_save_progress<M: LlmModelMarker>(
        &mut self,
        status: OrchestrationStatus,
        epoch: usize,
    ) {
        self.progress.status = status;
        self.progress.epoch = epoch;
        let progress_save_path =
            Orchestrator::progress_save_path(M::CLI_NAME.into(), &self.config_nickname);
        write_json(&progress_save_path, &self.progress).unwrap();
    }

    async fn read_and_log_validation_accuracy<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
    ) -> Result<(), String> {
        log_info("Reading and logging validation accuracy...");
        let asset_file_action_logs = AssetFileDirectTreeActionLogs::<M> {
            nickname: self.config_nickname.clone(),
            rollout_config: self.validation_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            _phantom: std::marker::PhantomData,
        };
        let win_rate = read_accuracy(asset_file_action_logs).await;
        if win_rate.total_plays == 0 {
            return Err("Validation action log is empty, cannot compute accuracy".to_string());
        }
        let accuracy = win_rate.num_wins as f32 / win_rate.total_plays as f32;
        self.progress.validation_accuracies.insert(epoch, accuracy);
        log_key_value_pair(
            format!("epoch_{}_start_accuracy", epoch),
            accuracy.to_string(),
        );
        let progress_save_path =
            Orchestrator::progress_save_path(M::CLI_NAME.into(), &self.config_nickname);
        write_json(&progress_save_path, &self.progress).unwrap();
        log_info(format!(
            "Epoch {} validation accuracy: {} ({} wins out of {} plays)",
            epoch, accuracy, win_rate.num_wins, win_rate.total_plays
        ));
        Ok(())
    }

    async fn ensure_inference_server_launched<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
    ) -> Result<(), String> {
        if let Some(handle) = &self.inference_server_handle {
            log_info(format!(
                "ensure_inference_server_launched: found existing handle (stored_epoch={}, has_port={}, has_process={}) while requesting epoch {}",
                handle.epoch,
                handle.sglang_port.is_some(),
                handle.process.is_some(),
                epoch,
            ));
            if handle.epoch == epoch {
                // already launched for this epoch
                log_info(format!(
                    "ensure_inference_server_launched: reusing existing inference server handle for epoch {}",
                    epoch
                ));
                return Ok(());
            } else {
                // first shut down the previous one
                log_info(format!(
                    "ensure_inference_server_launched: existing handle epoch {} differs from requested epoch {}, shutting down first",
                    handle.epoch, epoch
                ));
                self.ensure_inference_server_shut_down().await;
                // then continue to launch the new one
                self.launch_inference_server::<M>(epoch).await?;
            }
        } else {
            // not launched, just launch
            log_info(format!(
                "ensure_inference_server_launched: no existing handle for requested epoch {}, launching new server",
                epoch
            ));
            self.launch_inference_server::<M>(epoch).await?;
        }
        Ok(())
    }

    async fn launch_inference_server<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
    ) -> Result<(), String> {
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
            return Ok(());
        }

        if epoch == 0 {
            crate::load_initial_model::load_initial_model(&self.epoch_dir::<M>(epoch), M::API_NAME)
                .await?;
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
        Ok(())
    }

    async fn ensure_inference_server_shut_down(&mut self) {
        if let Some(handle) = self.inference_server_handle.take() {
            log_info(format!(
                "Shutting down inference server (stored_epoch={}, has_port={}, has_process={})...",
                handle.epoch,
                handle.sglang_port.is_some(),
                handle.process.is_some(),
            ));
            if let Some(mut process) = handle.process {
                let pid_for_log = process.id();
                match process.try_wait() {
                    Ok(Some(status)) => log_info(format!(
                        "Inference server process already exited before shutdown (pid={:?}, status={})",
                        pid_for_log, status
                    )),
                    Ok(None) => log_info(format!(
                        "Inference server process appears alive before shutdown (pid={:?})",
                        pid_for_log
                    )),
                    Err(err) => log_info(format!(
                        "Failed to probe inference server process status before shutdown (pid={:?}): {}",
                        pid_for_log, err
                    )),
                }
                shut_down_sglang_server_process(&mut process).await;
                log_info(format!(
                    "Completed shutdown call for inference server process (pid={:?})",
                    pid_for_log
                ));
            } else {
                log_info("Inference server handle had no process to shut down");
            }
            log_info("Inference server shut down");
        } else {
            log_info(
                "ensure_inference_server_shut_down: no inference server handle present; nothing to shut down",
            );
        }
        assert!(
            self.inference_server_handle.is_none(),
            "Inference server should be shut down, but it's still Some"
        );
    }

    async fn validate_model<M: LlmModelMarker>(&self, epoch: usize) -> Result<(), String> {
        log_info("Start validating model.");
        assert!(self.validation_rollout_config.split == DatasetSplit::Validation);

        let validation_rollout_program_config = RolloutProgramConfig {
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.validation_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            client: self.client.clone(),
            max_rollout_concurrency: self.max_rollout_concurrency,
            llm_cli_args: LlmCliArgs {
                sglang_port: self
                    .inference_server_handle
                    .as_ref()
                    .and_then(|handle| handle.sglang_port),
            },
            rollout_time_limit_secs: self.validation_rollout_time_limit_secs,
            num_python_tool_servers: self.num_python_tool_servers,
        };
        rollout_all::<M>(validation_rollout_program_config).await;
        log_info("Finished validating model.");
        Ok(())
    }

    async fn collect_training_rollout<M: LlmModelMarker>(&self, epoch: usize) {
        log_info("Collecting training rollout");
        let Some(sglang_server_handle) = &self.inference_server_handle else {
            panic!("Orchestrator did not launch the sglang server before generating training set");
        };
        let llm_cli_args = LlmCliArgs {
            sglang_port: sglang_server_handle.sglang_port.clone(),
        };
        assert!(self.training_set_rollout_config.split == DatasetSplit::Training);
        let training_set_rollout_program_config = RolloutProgramConfig {
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            client: self.client.clone(),
            max_rollout_concurrency: self.max_rollout_concurrency,
            llm_cli_args,
            rollout_time_limit_secs: self.training_rollout_time_limit_secs,
            num_python_tool_servers: self.num_python_tool_servers,
        };
        rollout_all::<M>(training_set_rollout_program_config).await;
        log_info("Finished collecting training rollout");
    }

    async fn generate_training_set<M: LlmModelMarker>(&self, epoch: usize) {
        log_info("Generating training set");
        let asset_file_training_trajectories = AssetFileTrainingTrajectories {
            config_nickname: self.config_nickname.clone(),
            epoch,
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            cumulative_avg_abs_advantage_cutoff: self.cumulative_avg_abs_advantage_cutoff,
            advantage_calculation_policy: self.advantage_calculation_policy,
            _phantom: std::marker::PhantomData::<M>,
        };
        asset_file_training_trajectories.synchronize().await;
        log_info("Finished generating training set");
    }
    async fn train_model<M: LlmModelMarker>(&self, epoch: usize) -> Result<(), String> {
        log_info("Start training model.");
        log_info(format!(
            "train_model called for epoch {} with in-memory inference_server_handle_present={}",
            epoch,
            self.inference_server_handle.is_some()
        ));
        let asset_file_training_trajectories = AssetFileTrainingTrajectories {
            config_nickname: self.config_nickname.clone(),
            epoch,
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            cumulative_avg_abs_advantage_cutoff: self.cumulative_avg_abs_advantage_cutoff,
            advantage_calculation_policy: self.advantage_calculation_policy,
            _phantom: std::marker::PhantomData::<M>,
        };
        let training_trajectory_sqlite_path = asset_file_training_trajectories.file_path();
        // first we need to write the training config to the expected location
        let training_config = PythonTrainingConfig {
            common: self.training_config_common.clone(),
            training_time: self.training_time,
            num_iterations_limit: self.num_iterations_limit,
            model_parent_dir: self.epoch_dir::<M>(epoch),
            training_trajectory_sqlite_path,
            checkpoints_parent_dir: self.epoch_dir::<M>(epoch),
            final_model_output_parent_dir: self.epoch_dir::<M>(epoch + 1),
        };
        let training_config_path = self.python_training_config_path::<M>(epoch);
        write_toml(&training_config_path, &training_config).unwrap();
        // launch the training python code
        let mut handle = launch_python_training_process(self.num_gpus, training_config_path).await;
        // wait for it to finish, do not interrupt unless the process crashes
        let exit_status = handle
            .process
            .wait()
            .await
            .map_err(|err| format!("Failed to wait for python training process: {}", err))?;
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
        log_info("Finished training model.");
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
            "results/{}/{}/epoch_{}/python_training_config.toml",
            M::CLI_NAME,
            self.config_nickname,
            epoch
        )
    }
}
impl Drop for Orchestrator {
    fn drop(&mut self) {
        if let Some(handle) = self.inference_server_handle.as_mut() {
            if let Some(process) = handle.process.as_mut() {
                let _ = process.start_kill();
            }
        }
        self.inference_server_handle = None;
        log_info("Orchestrator dropped, inference server (if any) should be shut down");
        println!("Orchestrator dropped, inference server (if any) should be shut down");
    }
}
