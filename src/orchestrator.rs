use research_utility::progress_text_logger::{
    log_info, log_key_value_pair, log_state, log_warning,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, time::Instant};

use ordered_float::NotNan;

use crate::{
    config_paths::{ConfigPaths, config_paths_file_path, derive_testing_rollout_config_path},
    constants,
    constants::get_max_concurrent_rollout,
    directories::{
        base_model_dir, model_metrics_path, model_parent_dir, progress_save_path,
        training_summary_parent_dir,
    },
    get_accuracy::get_accuracy,
    hybrid_dataset::{Training, Validation},
    json_toml_utils::write_json,
    launch_inference_wrapper::{
        best_effort_shutdown_stale_inference_wrapper, launch_inference_wrapper_process,
        shut_down_inference_wrapper_process,
    },
    launch_training_wrapper::run_training_wrapper_and_wait,
    llm_model::{InferenceEndpoint, LlmModelMarker},
    posterior_calculation_config::PosteriorCalculationConfig,
    python_training_config::{PythonTrainingConfig, TrainingHyperparameters, TrainingMode},
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::{RolloutConfig, TrainingRolloutConfig},
    training_set::{
        TrainingSetSortMode, generate_training_trajectories, open_training_trajectories,
        training_trajectories_file_path, training_trajectories_msgpack_file_path,
        training_trajectories_stats_file_path,
    },
    tree_action_log::action_logs_file_path,
};
use research_utility::launch_python_process::PythonProcessHandle;

pub struct Orchestrator {
    // for rollout
    pub config_nickname: String,
    pub validation_rollout_config: RolloutConfig<Validation>,
    pub training_set_rollout_config: TrainingRolloutConfig,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub training_rollout_secs: usize,
    pub validation_rollout_secs: usize,
    pub inference_server_handle: Option<InferenceServerHandle>,
    pub inference_wrapper_log_path: String,
    pub training_wrapper_log_path: String,
    pub keep_action_logs: bool,
    // for training set generation
    pub positive_advantage_only: bool,
    // for orchestration
    pub num_total_epochs: usize,
    // utilities
    pub client: reqwest::Client,
    // state
    pub progress: OrchestrationProgress,
    // for training
    pub training_hyperparameters: TrainingHyperparameters,
    pub training_time: f32,
    pub num_iterations_limit: usize,
    pub num_gpus: usize,
    pub use_tool: bool,
    pub mount_dir: String,
    pub training_set_sort_mode: TrainingSetSortMode,
}

pub struct InferenceServerHandle {
    pub epoch: usize,
    pub use_tool: bool,
    pub inference_endpoint: InferenceEndpoint,
    pub wrapper_handle: Option<PythonProcessHandle>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OrchestrationStatus {
    WorkingOnValidation,
    WorkingOnTrainingRolloutCollection,
    WorkingOnTrainingSetGeneration,
    WorkingOnTraining,
    Completed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrchestrationProgress {
    pub status: OrchestrationStatus,
    pub epoch: usize,
    pub validation_accuracies: BTreeMap<usize, (f32, f32, f32, f32)>,
    pub training_rollout_accuracies: BTreeMap<usize, (f32, f32, f32, f32)>,
    #[serde(default)]
    pub validation_rollout_llm_call_throughputs: BTreeMap<usize, f32>,
    #[serde(default)]
    pub training_rollout_llm_call_throughputs: BTreeMap<usize, f32>,
    #[serde(default)]
    pub training_throughputs: BTreeMap<usize, f32>,
}

fn accuracy_tuple_to_string(accuracies: (f32, f32, f32, f32)) -> String {
    format!(
        "({:.6}, {:.6}, {:.6}, {:.6})",
        accuracies.0, accuracies.1, accuracies.2, accuracies.3
    )
}

fn average_accuracy(accuracies: (f32, f32, f32, f32)) -> f32 {
    accuracies.0
}

impl Orchestrator {
    fn ensure_inference_server_process_alive(&mut self, context: &str) -> Result<(), String> {
        enum Probe {
            Alive,
            Exited {
                pid: Option<u32>,
                status: std::process::ExitStatus,
            },
            ProbeError {
                pid: Option<u32>,
                message: String,
            },
        }

        let probe = {
            let Some(handle) = self.inference_server_handle.as_mut() else {
                return Ok(());
            };
            let Some(ref mut wrapper) = handle.wrapper_handle else {
                return Ok(());
            };
            let pid_for_log = wrapper.child.id();
            match wrapper.child.try_wait() {
                Ok(Some(status)) => Probe::Exited {
                    pid: pid_for_log,
                    status,
                },
                Ok(None) => Probe::Alive,
                Err(err) => Probe::ProbeError {
                    pid: pid_for_log,
                    message: err.to_string(),
                },
            }
        };

        match probe {
            Probe::Alive => Ok(()),
            Probe::Exited { pid, status } => {
                if let Some(handle) = self.inference_server_handle.as_ref() {
                    if let Some(ref wrapper) = handle.wrapper_handle {
                        let _ = wrapper.stop_signal_tx.send(true);
                    }
                }
                self.inference_server_handle = None;
                Err(format!(
                    "Inference server process exited unexpectedly {} (pid={:?}, status={})",
                    context, pid, status
                ))
            }
            Probe::ProbeError { pid, message } => Err(format!(
                "Failed to probe inference server process status {} (pid={:?}): {}",
                context, pid, message
            )),
        }
    }

    pub fn progress_save_path(
        mount_dir: &str,
        model_cli_name: &str,
        config_nickname: &str,
    ) -> String {
        progress_save_path(mount_dir, model_cli_name, config_nickname)
    }

    pub fn write_config_paths_file(
        model_cli_name: &str,
        config_nickname: &str,
        training_rollout_config_path: &str,
        validation_rollout_config_path: &str,
    ) -> Result<(), String> {
        let testing_rollout_config_path =
            derive_testing_rollout_config_path(validation_rollout_config_path)?;
        let config_paths = ConfigPaths {
            training_rollout_config_path: training_rollout_config_path.to_string(),
            validation_rollout_config_path: validation_rollout_config_path.to_string(),
            testing_rollout_config_path,
        };
        let config_paths_path = config_paths_file_path(model_cli_name, config_nickname)?;
        write_json(config_paths_path, &config_paths)
    }

    pub async fn orchestrate<M: LlmModelMarker>(&mut self) -> Result<(), String> {
        assert!(
            self.num_total_epochs > 0,
            "num_total_epochs must be positive for orchestration"
        );
        loop {
            let progress = self.progress.clone();
            let epoch = progress.epoch;
            match progress.status {
                OrchestrationStatus::WorkingOnValidation => {
                    log_state(format!("Epoch {}: Working on validation", epoch));
                    assert!(epoch <= self.num_total_epochs);
                    self.ensure_inference_server_launched::<M>(epoch, self.use_tool)
                        .await?;
                    self.validate_model::<M>(epoch).await?;
                    self.read_and_log_validation_accuracy::<M>(epoch).await?;
                    self.sweep_previous_model_dirs_after_validation::<M>(epoch)?;
                    if epoch >= self.num_total_epochs {
                        log_info(&format!(
                            "Finished all {} epochs of orchestration",
                            self.num_total_epochs
                        ));
                        self.ensure_inference_server_shut_down::<M>().await;
                        self.cleanup_epoch_model_dir_if_not_best::<M>(epoch)?;
                        self.update_and_save_progress::<M>(OrchestrationStatus::Completed, epoch);
                        break;
                    }
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnTrainingRolloutCollection,
                        epoch,
                    );
                }
                OrchestrationStatus::WorkingOnTrainingRolloutCollection => {
                    log_state(format!(
                        "Epoch {}: Working on training rollout collection",
                        epoch
                    ));
                    self.ensure_inference_server_launched::<M>(epoch, self.use_tool)
                        .await?;
                    self.collect_training_rollout::<M>(epoch).await?;
                    self.read_and_log_training_rollout_accuracy::<M>(epoch)
                        .await?;
                    // after rollout collection, we can shut down the inference server
                    self.ensure_inference_server_shut_down::<M>().await;
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnTrainingSetGeneration,
                        epoch,
                    );
                }
                OrchestrationStatus::WorkingOnTrainingSetGeneration => {
                    log_state(format!(
                        "Epoch {}: Working on training set generation",
                        epoch
                    ));
                    // we do not need inference server for training set generation, and it won't be launched again until we do the training step
                    self.ensure_inference_server_shut_down::<M>().await;
                    self.generate_training_set::<M>(epoch).await;
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnTraining,
                        epoch,
                    );
                }
                OrchestrationStatus::WorkingOnTraining => {
                    // we do not want inference server to be up during training
                    log_state(format!("Epoch {}: Working on training", epoch));
                    log_info(format!(
                        "Entering training stage for epoch {}. About to enforce inference-server shutdown.",
                        epoch
                    ));
                    self.ensure_inference_server_shut_down::<M>().await;
                    log_info(format!(
                        "Inference-server shutdown enforcement finished for epoch {}. Starting training.",
                        epoch
                    ));
                    self.train_model::<M>(epoch).await?;
                    self.cleanup_epoch_artifacts_after_training::<M>(epoch)?;
                    assert!(epoch < self.num_total_epochs);
                    // do the final validation
                    self.update_and_save_progress::<M>(
                        OrchestrationStatus::WorkingOnValidation,
                        epoch + 1,
                    );
                }
                OrchestrationStatus::Completed => {
                    log_info(format!(
                        "Orchestration already completed at epoch {}; exiting without additional work",
                        epoch
                    ));
                    break;
                }
            }
        }
        log_state("All training finished");
        // for safety
        self.ensure_inference_server_shut_down::<M>().await;
        Ok(())
    }

    fn save_progress<M: LlmModelMarker>(&self) {
        let progress_save_path = Orchestrator::progress_save_path(
            &self.mount_dir,
            M::CLI_NAME.into(),
            &self.config_nickname,
        );
        write_json(&progress_save_path, &self.progress).unwrap();
    }

    fn update_and_save_progress<M: LlmModelMarker>(
        &mut self,
        status: OrchestrationStatus,
        epoch: usize,
    ) {
        self.progress.status = status;
        self.progress.epoch = epoch;
        self.save_progress::<M>();
    }

    async fn read_and_log_validation_accuracy<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
    ) -> Result<(), String> {
        log_info("Reading and logging validation accuracy...");
        let accuracy_stats = get_accuracy::<M, Validation>(
            &self.mount_dir,
            self.config_nickname.clone(),
            self.validation_rollout_config.clone(),
            self.posterior_calculation_config.clone(),
            epoch,
            "Validation accuracy",
            self.use_tool,
        )
        .await;
        let Some(accuracies) = accuracy_stats.accuracy_tuple() else {
            return Err("Validation action log is empty, cannot compute accuracy".to_string());
        };
        self.progress
            .validation_accuracies
            .insert(epoch, accuracies);
        log_key_value_pair(
            format!("epoch_{}_validation_accuracy_deepmath", epoch),
            accuracies.1.to_string(),
        );
        log_key_value_pair(
            format!("epoch_{}_validation_accuracy_math", epoch),
            accuracies.2.to_string(),
        );
        log_key_value_pair(
            format!("epoch_{}_validation_accuracy_numinamath", epoch),
            accuracies.3.to_string(),
        );
        self.save_progress::<M>();
        log_info(format!(
            "Epoch {} validation accuracies (avg, deepmath, math, numinamath): {} (avg {:.6}, weighted wins {:.4} over {} trees, {} trajectories)",
            epoch,
            accuracy_tuple_to_string(accuracies),
            average_accuracy(accuracies),
            accuracy_stats.weighted_num_wins,
            accuracy_stats.num_trees_with_judgments,
            accuracy_stats.num_trajectories_judged
        ));
        Ok(())
    }

    async fn read_and_log_training_rollout_accuracy<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
    ) -> Result<(), String> {
        log_info("Reading and logging training rollout accuracy...");
        let accuracy_stats = get_accuracy::<M, Training>(
            &self.mount_dir,
            self.config_nickname.clone(),
            self.training_set_rollout_config.to_rollout_config(),
            self.posterior_calculation_config.clone(),
            epoch,
            "Training rollout accuracy",
            self.use_tool,
        )
        .await;
        let Some(accuracies) = accuracy_stats.accuracy_tuple() else {
            return Err(
                "Training rollout action log is empty, cannot compute accuracy".to_string(),
            );
        };
        self.progress
            .training_rollout_accuracies
            .insert(epoch, accuracies);
        log_key_value_pair(
            format!("epoch_{}_training_rollout_accuracy_deepmath", epoch),
            accuracies.1.to_string(),
        );
        log_key_value_pair(
            format!("epoch_{}_training_rollout_accuracy_math", epoch),
            accuracies.2.to_string(),
        );
        log_key_value_pair(
            format!("epoch_{}_training_rollout_accuracy_numinamath", epoch),
            accuracies.3.to_string(),
        );
        self.save_progress::<M>();
        log_info(format!(
            "Epoch {} training rollout accuracies (avg, deepmath, math, numinamath): {} (avg {:.6}, weighted wins {:.4} over {} trees, {} trajectories)",
            epoch,
            accuracy_tuple_to_string(accuracies),
            average_accuracy(accuracies),
            accuracy_stats.weighted_num_wins,
            accuracy_stats.num_trees_with_judgments,
            accuracy_stats.num_trajectories_judged
        ));
        Ok(())
    }

    async fn ensure_inference_server_launched<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
        use_tool: bool,
    ) -> Result<(), String> {
        self.ensure_inference_server_process_alive("before launch/reuse check")?;
        if let Some(handle) = &self.inference_server_handle {
            log_info(format!(
                "ensure_inference_server_launched: found existing handle (stored_epoch={}, stored_use_tool={}, endpoint={:?}, has_process={}) while requesting epoch {} use_tool={}",
                handle.epoch,
                handle.use_tool,
                handle.inference_endpoint,
                handle.wrapper_handle.is_some(),
                epoch,
                use_tool,
            ));
            assert!(
                handle.use_tool == use_tool,
                "inference server handle use_tool mismatch: existing={} requested={}",
                handle.use_tool,
                use_tool
            );
            if handle.epoch == epoch && handle.use_tool == use_tool {
                // already launched for this epoch
                self.ensure_inference_server_process_alive("while reusing existing handle")?;
                log_info(format!(
                    "ensure_inference_server_launched: reusing existing inference server handle for epoch {} use_tool={}",
                    epoch, use_tool
                ));
                return Ok(());
            } else {
                // first shut down the previous one
                log_info(format!(
                    "ensure_inference_server_launched: existing handle (epoch={}, use_tool={}) differs from requested (epoch={}, use_tool={}), shutting down first",
                    handle.epoch, handle.use_tool, epoch, use_tool
                ));
                self.ensure_inference_server_shut_down::<M>().await;
                // then continue to launch the new one
                self.launch_inference_server::<M>(epoch, use_tool).await?;
            }
        } else {
            // not launched, just launch
            log_info(format!(
                "ensure_inference_server_launched: no existing handle for requested epoch {} use_tool={}, launching new server",
                epoch, use_tool
            ));
            self.launch_inference_server::<M>(epoch, use_tool).await?;
        }
        Ok(())
    }

    async fn launch_inference_server<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
        use_tool: bool,
    ) -> Result<(), String> {
        assert!(
            self.inference_server_handle.is_none(),
            "Inference server is already launched for epoch {}, cannot launch again without shutting down",
            self.inference_server_handle.as_ref().unwrap().epoch
        );
        let model_parent_dir =
            model_parent_dir(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch);
        let model_path = format!("{}/model", model_parent_dir);

        log_info(format!(
            "Launching local inference wrapper for model {}",
            M::CLI_NAME,
        ));
        let (sglang_port, handle) = launch_inference_wrapper_process(
            model_path.as_ref(),
            M::CLI_NAME,
            &self.config_nickname,
            epoch,
            M::API_NAME,
            self.num_gpus,
            self.inference_wrapper_log_path.as_ref(),
        )
        .await?;
        log_info(format!(
            "Inference wrapper is listening on port {}",
            sglang_port
        ));

        self.inference_server_handle = Some(InferenceServerHandle {
            epoch,
            use_tool,
            inference_endpoint: InferenceEndpoint::SglangPort(sglang_port),
            wrapper_handle: Some(handle),
        });
        log_info("Inference wrapper launched");
        Ok(())
    }

    async fn ensure_inference_server_shut_down<M: LlmModelMarker>(&mut self) {
        if let Some(handle) = self.inference_server_handle.take() {
            let InferenceServerHandle {
                epoch,
                use_tool,
                inference_endpoint,
                wrapper_handle,
            } = handle;
            log_info(format!(
                "Shutting down inference server (stored_epoch={}, stored_use_tool={}, endpoint={:?}, has_process={})...",
                epoch,
                use_tool,
                inference_endpoint,
                wrapper_handle.is_some(),
            ));
            if let Some(mut proc_handle) = wrapper_handle {
                let pid_for_log = proc_handle.child.id();
                let _ = proc_handle.stop_signal_tx.send(true);
                match proc_handle.child.try_wait() {
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
                shut_down_inference_wrapper_process(&mut proc_handle.child).await;
                log_info(format!(
                    "Completed shutdown call for inference server process (pid={:?})",
                    pid_for_log
                ));
                match proc_handle.listener_handle.await {
                    Ok(()) => log_info("Inference server TUI listener joined successfully"),
                    Err(err) => log_warning(format!(
                        "Inference server TUI listener join failed: {}",
                        err
                    )),
                }
            } else {
                log_info("Inference server handle had no process to shut down");
            }
            log_info("Inference server shut down");
        } else {
            log_info(
                "ensure_inference_server_shut_down: no inference server handle present; checking for stale local sglang process on configured port",
            );
            best_effort_shutdown_stale_inference_wrapper().await;
        }
        assert!(
            self.inference_server_handle.is_none(),
            "Inference server should be shut down, but it's still Some"
        );
    }

    async fn validate_model<M: LlmModelMarker>(&mut self, epoch: usize) -> Result<(), String> {
        log_info("Start validating model.");
        self.ensure_inference_server_process_alive("before validation rollout")?;
        // assert!(self.validation_rollout_config.split == DatasetSplit::Validation);

        let validation_rollout_program_config = RolloutProgramConfig {
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.validation_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            client: self.client.clone(),
            inference_endpoint: self
                .inference_server_handle
                .as_ref()
                .map(|handle| handle.inference_endpoint.clone())
                .expect(
                    "Orchestrator did not launch the inference server before validation rollout",
                ),
            rollout_secs: self.validation_rollout_secs,
            total_epochs: self.num_total_epochs,
            action_log_store_override_path: None,
            use_tool: self.use_tool,
            fixed_temperature: NotNan::new(constants::VALIDATION_TEMPERATURE).unwrap(),
            max_concurrent_rollout: get_max_concurrent_rollout(self.num_gpus),
        };
        let rollout_summary =
            rollout_all::<M, Validation>(&self.mount_dir, validation_rollout_program_config).await;
        self.progress
            .validation_rollout_llm_call_throughputs
            .insert(epoch, rollout_summary.llm_call_throughput_per_sec);
        self.save_progress::<M>();
        log_key_value_pair(
            format!("epoch_{}_validation_llm_call_throughput_per_sec", epoch),
            format!("{:.6}", rollout_summary.llm_call_throughput_per_sec),
        );
        log_info(format!(
            "Epoch {} validation rollout LLM throughput: {:.6} calls/sec over {:.3}s ({} total LLM calls)",
            epoch,
            rollout_summary.llm_call_throughput_per_sec,
            rollout_summary.elapsed_secs,
            rollout_summary.total_llm_calls,
        ));
        self.ensure_inference_server_process_alive("after validation rollout")?;
        log_info("Finished validating model.");
        Ok(())
    }

    async fn collect_training_rollout<M: LlmModelMarker>(
        &mut self,
        epoch: usize,
    ) -> Result<(), String> {
        log_info("Collecting training rollout");
        self.ensure_inference_server_process_alive("before training rollout collection")?;
        self.remove_epoch_training_checkpoints::<M>(epoch)?;
        let Some(sglang_server_handle) = &self.inference_server_handle else {
            panic!("Orchestrator did not launch the sglang server before generating training set");
        };
        let inference_endpoint = sglang_server_handle.inference_endpoint.clone();
        // assert!(self.training_set_rollout_config.split == DatasetSplit::Training);
        let training_set_rollout_program_config = RolloutProgramConfig {
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.training_set_rollout_config.to_rollout_config(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            client: self.client.clone(),
            inference_endpoint,
            rollout_secs: self.training_rollout_secs,
            total_epochs: self.num_total_epochs,
            action_log_store_override_path: None,
            use_tool: self.use_tool,
            fixed_temperature: NotNan::new(constants::TRAINING_TEMPERATURE).unwrap(),
            max_concurrent_rollout: get_max_concurrent_rollout(self.num_gpus),
        };
        let rollout_summary =
            rollout_all::<M, Training>(&self.mount_dir, training_set_rollout_program_config).await;
        self.progress
            .training_rollout_llm_call_throughputs
            .insert(epoch, rollout_summary.llm_call_throughput_per_sec);
        self.save_progress::<M>();
        log_key_value_pair(
            format!(
                "epoch_{}_training_rollout_llm_call_throughput_per_sec",
                epoch
            ),
            format!("{:.6}", rollout_summary.llm_call_throughput_per_sec),
        );
        log_info(format!(
            "Epoch {} training rollout LLM throughput: {:.6} calls/sec over {:.3}s ({} total LLM calls)",
            epoch,
            rollout_summary.llm_call_throughput_per_sec,
            rollout_summary.elapsed_secs,
            rollout_summary.total_llm_calls,
        ));
        self.ensure_inference_server_process_alive("after training rollout collection")?;
        log_info("Finished collecting training rollout");
        Ok(())
    }

    fn remove_epoch_training_checkpoints<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        let checkpoint_parent_dir =
            model_parent_dir(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch);
        let metrics_path =
            model_metrics_path(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch);
        let checkpoints_dir = format!("{}/checkpoints", checkpoint_parent_dir);
        let latest_checkpoint_path = format!("{}/latest_checkpoint.txt", checkpoint_parent_dir);

        if Path::new(&checkpoints_dir).exists() {
            std::fs::remove_dir_all(&checkpoints_dir).map_err(|err| {
                format!(
                    "Failed to remove checkpoint directory for epoch {} ({}): {}",
                    epoch, checkpoints_dir, err
                )
            })?;
            log_info(format!(
                "Removed stale checkpoint directory for epoch {}: {}",
                epoch, checkpoints_dir
            ));
        }

        if Path::new(&latest_checkpoint_path).exists() {
            std::fs::remove_file(&latest_checkpoint_path).map_err(|err| {
                format!(
                    "Failed to remove latest checkpoint pointer for epoch {} ({}): {}",
                    epoch, latest_checkpoint_path, err
                )
            })?;
            log_info(format!(
                "Removed stale latest checkpoint pointer for epoch {}: {}",
                epoch, latest_checkpoint_path
            ));
        }

        if Path::new(&metrics_path).exists() {
            std::fs::remove_file(&metrics_path).map_err(|err| {
                format!(
                    "Failed to remove training metrics file for epoch {} ({}): {}",
                    epoch, metrics_path, err
                )
            })?;
            log_info(format!(
                "Removed stale training metrics file for epoch {}: {}",
                epoch, metrics_path
            ));
        }

        Ok(())
    }

    fn cleanup_epoch_artifacts_after_training<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        self.destroy_epoch_checkpoint_folder::<M>(epoch)?;
        self.destroy_epoch_rollout_action_logs::<M>(epoch)?;
        self.destroy_previous_epoch_training_trajectories::<M>(epoch)?;
        self.cleanup_epoch_model_dir_if_not_best::<M>(epoch)?;
        Ok(())
    }

    fn destroy_epoch_checkpoint_folder<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        let checkpoint_parent_dir =
            model_parent_dir(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch);
        let checkpoints_dir = format!("{}/checkpoints", checkpoint_parent_dir);
        let latest_checkpoint_path = format!("{}/latest_checkpoint.txt", checkpoint_parent_dir);

        if Path::new(&checkpoints_dir).exists() {
            std::fs::remove_dir_all(&checkpoints_dir).map_err(|err| {
                format!(
                    "Failed to remove checkpoint directory for epoch {} ({}): {}",
                    epoch, checkpoints_dir, err
                )
            })?;
            log_info(format!(
                "Removed checkpoint directory after training for epoch {}: {}",
                epoch, checkpoints_dir
            ));
        }

        if Path::new(&latest_checkpoint_path).exists() {
            std::fs::remove_file(&latest_checkpoint_path).map_err(|err| {
                format!(
                    "Failed to remove latest checkpoint pointer for epoch {} ({}): {}",
                    epoch, latest_checkpoint_path, err
                )
            })?;
            log_info(format!(
                "Removed latest checkpoint pointer after training for epoch {}: {}",
                epoch, latest_checkpoint_path
            ));
        }

        Ok(())
    }

    fn destroy_epoch_rollout_action_logs<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        if self.keep_action_logs {
            log_info(format!(
                "Keeping training and validation rollout action logs for epoch {}",
                epoch
            ));
            return Ok(());
        }

        self.delete_file_if_exists(
            &action_logs_file_path::<M, Training>(&self.mount_dir, &self.config_nickname, epoch),
            &format!("training rollout action logs for epoch {}", epoch),
        )?;

        self.delete_file_if_exists(
            &action_logs_file_path::<M, Validation>(&self.mount_dir, &self.config_nickname, epoch),
            &format!("validation rollout action logs for epoch {}", epoch),
        )?;

        Ok(())
    }

    fn destroy_previous_epoch_training_trajectories<M: LlmModelMarker>(
        &self,
        current_epoch: usize,
    ) -> Result<(), String> {
        if current_epoch <= 1 {
            log_info(format!(
                "Keeping training trajectories for epoch 0 and latest epoch {}",
                current_epoch
            ));
            return Ok(());
        }

        for epoch in 1..current_epoch {
            self.delete_dir_if_exists(
                &training_trajectories_file_path::<M>(
                    &self.mount_dir,
                    &self.config_nickname,
                    epoch,
                ),
                &format!("training trajectories directory for epoch {}", epoch),
            )?;
            self.delete_file_if_exists(
                &training_trajectories_stats_file_path::<M>(
                    &self.mount_dir,
                    &self.config_nickname,
                    epoch,
                ),
                &format!("training trajectories stats for epoch {}", epoch),
            )?;
        }

        Ok(())
    }

    fn cleanup_epoch_model_dir_if_not_best<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        let Some(&epoch_accuracies) = self.progress.validation_accuracies.get(&epoch) else {
            return Err(format!(
                "Missing validation accuracy for epoch {}, cannot decide model cleanup",
                epoch
            ));
        };
        let epoch_avg_accuracy = average_accuracy(epoch_accuracies);
        let max_avg_accuracy_through_epoch = self
            .progress
            .validation_accuracies
            .iter()
            .filter(|(logged_epoch, _)| **logged_epoch <= epoch)
            .map(|(_, accuracies)| average_accuracy(*accuracies))
            .max_by(|a, b| a.total_cmp(b))
            .ok_or_else(|| {
                format!(
                    "No validation accuracy recorded through epoch {}, cannot decide model cleanup",
                    epoch
                )
            })?;
        let keep_model_dir = epoch_avg_accuracy >= max_avg_accuracy_through_epoch;
        if keep_model_dir {
            log_info(format!(
                "Keeping model directory for epoch {} because validation average accuracy {:.6} from tuple {} is best so far",
                epoch,
                epoch_avg_accuracy,
                accuracy_tuple_to_string(epoch_accuracies)
            ));
            return Ok(());
        }

        if epoch == 0 {
            log_info(
                "Skipping epoch 0 model directory deletion because epoch 0 parent path is a shared root",
            );
            return Ok(());
        }

        let model_parent_dir =
            model_parent_dir(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch);
        self.delete_dir_if_exists(
            &model_parent_dir,
            &format!("model parent directory for epoch {}", epoch),
        )
    }

    fn sweep_previous_model_dirs_after_validation<M: LlmModelMarker>(
        &self,
        current_epoch: usize,
    ) -> Result<(), String> {
        let best_epoch_so_far = self
            .progress
            .validation_accuracies
            .iter()
            .filter(|(epoch, _)| **epoch <= current_epoch)
            .max_by(|a, b| {
                average_accuracy(*a.1)
                    .total_cmp(&average_accuracy(*b.1))
                    .then_with(|| a.0.cmp(b.0))
            })
            .map(|(epoch, _)| *epoch)
            .ok_or_else(|| {
                format!(
                    "No validation accuracies recorded up to epoch {}, cannot sweep model directories",
                    current_epoch
                )
            })?;

        for epoch in self.progress.validation_accuracies.keys().copied() {
            if epoch >= current_epoch || epoch == best_epoch_so_far {
                continue;
            }
            if epoch == 0 {
                log_info(
                    "Skipping epoch 0 model directory deletion during sweep because epoch 0 parent path is a shared root",
                );
                continue;
            }
            let model_parent_dir =
                model_parent_dir(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch);
            self.delete_dir_if_exists(
                &model_parent_dir,
                &format!(
                    "non-best previous model parent directory for epoch {}",
                    epoch
                ),
            )?;
        }
        Ok(())
    }

    fn delete_file_if_exists(&self, path: &str, description: &str) -> Result<(), String> {
        if Path::new(path).exists() {
            std::fs::remove_file(path)
                .map_err(|err| format!("Failed to remove {} ({}): {}", description, path, err))?;
            log_info(format!("Removed {}: {}", description, path));
        }
        Ok(())
    }

    fn delete_dir_if_exists(&self, path: &str, description: &str) -> Result<(), String> {
        if Path::new(path).exists() {
            std::fs::remove_dir_all(path)
                .map_err(|err| format!("Failed to remove {} ({}): {}", description, path, err))?;
            log_info(format!("Removed {}: {}", description, path));
        }
        Ok(())
    }

    async fn generate_training_set<M: LlmModelMarker>(&self, epoch: usize) {
        log_info("Generating training set");
        generate_training_trajectories::<M>(
            &self.mount_dir,
            &self.config_nickname,
            self.training_set_rollout_config.to_rollout_config(),
            self.posterior_calculation_config.clone(),
            epoch,
            self.training_set_rollout_config.training_advantage_policy,
            self.positive_advantage_only,
            self.use_tool,
            self.training_set_sort_mode,
        )
        .await;
        log_info("Finished generating training set");
    }
    async fn train_model<M: LlmModelMarker>(&mut self, epoch: usize) -> Result<(), String> {
        log_info("Start training model.");
        log_info(format!(
            "train_model called for epoch {} with in-memory inference_server_handle_present={}",
            epoch,
            self.inference_server_handle.is_some()
        ));
        let training_trajectory_store =
            open_training_trajectories::<M>(&self.mount_dir, &self.config_nickname, epoch);
        let num_training_samples = training_trajectory_store.len();
        let training_trajectory_path = training_trajectories_msgpack_file_path::<M>(
            &self.mount_dir,
            &self.config_nickname,
            epoch,
        );
        let input_model_dir = if epoch == 0 {
            base_model_dir(&self.mount_dir, M::CLI_NAME)
        } else {
            model_parent_dir(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch)
        };
        let final_model_output_parent_dir = model_parent_dir(
            &self.mount_dir,
            M::CLI_NAME,
            &self.config_nickname,
            epoch + 1,
        );
        let training_summary_dir =
            training_summary_parent_dir(&self.mount_dir, M::CLI_NAME, &self.config_nickname, epoch);
        let training_config = PythonTrainingConfig {
            hyperparameters: self.training_hyperparameters.clone(),
            num_iterations_limit: self.num_iterations_limit,
            model_cli_name: M::CLI_NAME.to_string(),
            config_nickname: self.config_nickname.clone(),
            training_mode: TrainingMode::Orchestration {
                epoch,
                training_time: self.training_time,
                input_model_parent_dir: input_model_dir.clone(),
                output_model_parent_dir: final_model_output_parent_dir,
                training_summary_dir,
            },
        };

        let training_start_time = Instant::now();
        run_training_wrapper_and_wait(
            self.num_gpus,
            M::API_NAME,
            &training_config,
            &training_trajectory_path,
            self.training_wrapper_log_path.as_ref(),
        )
        .await?;
        let elapsed_secs = training_start_time.elapsed().as_secs_f32();
        let training_throughput = if elapsed_secs <= f32::EPSILON {
            0.0
        } else {
            num_training_samples as f32 / elapsed_secs
        };
        self.progress
            .training_throughputs
            .insert(epoch, training_throughput);
        self.save_progress::<M>();
        log_key_value_pair(
            format!("epoch_{}_training_throughput_num_samples_per_sec", epoch),
            format!("{training_throughput:.6}"),
        );
        log_info(format!(
            "Epoch {} training throughput: {:.6} samples/sec over {:.3}s ({} training samples)",
            epoch, training_throughput, elapsed_secs, num_training_samples,
        ));
        log_info("Finished training model.");
        Ok(())
    }
}
impl Drop for Orchestrator {
    fn drop(&mut self) {
        if let Some(handle) = self.inference_server_handle.as_mut() {
            if let Some(ref mut wrapper) = handle.wrapper_handle {
                let _ = wrapper.stop_signal_tx.send(true);
                let _ = wrapper.child.start_kill();
            }
        }
        self.inference_server_handle = None;
        log_info("Orchestrator dropped, inference server (if any) should be shut down");
        println!("Orchestrator dropped, inference server (if any) should be shut down");
    }
}
