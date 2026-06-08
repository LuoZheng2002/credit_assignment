use std::{collections::BTreeMap, path::Path, sync::LazyLock};

use minijinja::context;
use research_utility::{
    asset_file::AssetFile,
    progress_tui_logger::{log_info, log_key_value_pair, log_state},
};
use serde::{Deserialize, Serialize};
use tokio::process::Child;

use crate::{
    direct_tool::{
        direct_rollout::{RolloutProgramConfig, rollout_all},
        direct_rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
        direct_training_set::AssetFileTrainingTrajectories,
        direct_tree_action_log::AssetFileDirectTreeActionLogs,
        hybrid_dataset::{Training, Validation},
        posterior_calculation_config::PosteriorCalculationConfig,
    },
    get_accuracy::get_accuracy,
    json_line_util::{write_json, write_toml},
    launch_python_training::launch_python_training_process,
    launch_sglang_server::{
        best_effort_shutdown_stale_sglang_server, launch_sglang_server_process, model_uses_sglang,
        shut_down_sglang_server_process,
    },
    llm_model::{LlmCliArgs, LlmModelMarker},
    python_training_config::{PythonTrainingConfig, PythonTrainingConfigCommon},
    util::storage_dir_from_env,
};

pub const MODEL_PARENT_DIR_TEMPLATE_PATH: &str = "config/training/model_parent_dir.jinja";
pub const MODEL_CHECKPOINT_DIR_TEMPLATE_PATH: &str = "config/training/model_checkpoint_dir.jinja";
pub const MODEL_METRICS_PATH_TEMPLATE_PATH: &str = "config/training/model_metrics_path.jinja";
pub const TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_PATH: &str =
    "config/training/training_summary_parent_dir.jinja";

fn load_template_environment(
    template_path: &str,
    template_name: &'static str,
) -> Result<minijinja::Environment<'static>, String> {
    let template_source = std::fs::read_to_string(template_path)
        .map_err(|err| format!("Failed to read {}: {}", template_path, err))?;
    let mut env = minijinja::Environment::new();
    env.add_template_owned(template_name, template_source)
        .map_err(|err| format!("Failed to parse {} template: {}", template_name, err))?;
    Ok(env)
}

fn render_template_for_epoch(
    template_env: &LazyLock<Result<minijinja::Environment<'static>, String>>,
    template_name: &'static str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    let storage_dir = storage_dir_from_env()?;
    let env = template_env.as_ref().map_err(|err| err.clone())?;
    let template = env
        .get_template(template_name)
        .map_err(|err| format!("Failed to load {} template: {}", template_name, err))?;
    let rendered = template
        .render(context! {
            storage_dir => storage_dir,
            model_cli_name => model_cli_name,
            config_nickname => config_nickname,
            epoch => epoch,
        })
        .map_err(|err| format!("Failed to render {} template: {}", template_name, err))?;
    let rendered = rendered.trim().to_string();
    if rendered.is_empty() {
        return Err(format!("Rendered {} template is empty", template_name));
    }
    Ok(rendered)
}

pub static MODEL_PARENT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| load_template_environment(MODEL_PARENT_DIR_TEMPLATE_PATH, "model_parent_dir"));

pub static MODEL_CHECKPOINT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_template_environment(MODEL_CHECKPOINT_DIR_TEMPLATE_PATH, "model_checkpoint_dir")
});

pub static MODEL_METRICS_PATH_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_template_environment(MODEL_METRICS_PATH_TEMPLATE_PATH, "model_metrics_path")
});

pub static TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_ENVIRONMENT: LazyLock<
    Result<minijinja::Environment<'static>, String>,
> = LazyLock::new(|| {
    load_template_environment(
        TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_PATH,
        "training_summary_parent_dir",
    )
});

pub fn model_parent_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_template_for_epoch(
        &MODEL_PARENT_DIR_TEMPLATE_ENVIRONMENT,
        "model_parent_dir",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn model_checkpoint_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_template_for_epoch(
        &MODEL_CHECKPOINT_DIR_TEMPLATE_ENVIRONMENT,
        "model_checkpoint_dir",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn model_metrics_path_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_template_for_epoch(
        &MODEL_METRICS_PATH_TEMPLATE_ENVIRONMENT,
        "model_metrics_path",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub fn training_summary_parent_dir_from_template(
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) -> Result<String, String> {
    render_template_for_epoch(
        &TRAINING_SUMMARY_PARENT_DIR_TEMPLATE_ENVIRONMENT,
        "training_summary_parent_dir",
        model_cli_name,
        config_nickname,
        epoch,
    )
}

pub struct Orchestrator {
    // for rollout
    pub config_nickname: String,
    pub validation_rollout_config: DirectRolloutConfig<Validation>,
    pub training_set_rollout_config: DirectRolloutConfig<Training>,
    pub posterior_calculation_config: PosteriorCalculationConfig,
    pub training_rollout_time_limit_secs: usize,
    pub validation_rollout_time_limit_secs: usize,
    pub max_python_processes: usize,
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
    pub use_tool: bool,
    pub sglang_port: Option<u16>,
    pub process: Option<Child>,
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
                    log_state(format!("Epoch {}: Working on validation", epoch));
                    assert!(epoch <= self.num_total_epochs);
                    self.ensure_inference_server_launched::<M>(
                        epoch,
                        self.validation_rollout_config.use_tool,
                    )
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
                    self.ensure_inference_server_launched::<M>(
                        epoch,
                        self.training_set_rollout_config.use_tool,
                    )
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
        let asset_file_action_logs = AssetFileDirectTreeActionLogs::<M, Validation> {
            nickname: self.config_nickname.clone(),
            rollout_config: self.validation_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            _phantom: std::marker::PhantomData,
        };
        let accuracy_stats = get_accuracy(asset_file_action_logs, "Validation accuracy").await;
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
            format!("epoch_{}_validation_accuracy_gsm8k", epoch),
            accuracies.3.to_string(),
        );
        let progress_save_path =
            Orchestrator::progress_save_path(M::CLI_NAME.into(), &self.config_nickname);
        write_json(&progress_save_path, &self.progress).unwrap();
        log_info(format!(
            "Epoch {} validation accuracies (avg, deepmath, math, gsm8k): {} (avg {:.6}, weighted wins {:.4} over {} trees, {} trajectories)",
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
        let asset_file_action_logs = AssetFileDirectTreeActionLogs::<M, Training> {
            nickname: self.config_nickname.clone(),
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            _phantom: std::marker::PhantomData,
        };
        let accuracy_stats =
            get_accuracy(asset_file_action_logs, "Training rollout accuracy").await;
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
            format!("epoch_{}_training_rollout_accuracy_gsm8k", epoch),
            accuracies.3.to_string(),
        );
        let progress_save_path =
            Orchestrator::progress_save_path(M::CLI_NAME.into(), &self.config_nickname);
        write_json(&progress_save_path, &self.progress).unwrap();
        log_info(format!(
            "Epoch {} training rollout accuracies (avg, deepmath, math, gsm8k): {} (avg {:.6}, weighted wins {:.4} over {} trees, {} trajectories)",
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
        if let Some(handle) = &self.inference_server_handle {
            log_info(format!(
                "ensure_inference_server_launched: found existing handle (stored_epoch={}, stored_use_tool={}, has_port={}, has_process={}) while requesting epoch {} use_tool={}",
                handle.epoch,
                handle.use_tool,
                handle.sglang_port.is_some(),
                handle.process.is_some(),
                epoch,
                use_tool,
            ));
            if handle.epoch == epoch && handle.use_tool == use_tool {
                // already launched for this epoch
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
        if !model_uses_sglang::<M>() {
            self.inference_server_handle = Some(InferenceServerHandle {
                epoch,
                use_tool,
                sglang_port: None,
                process: None,
            });
            log_info(format!(
                "Model {} does not need a local inference server",
                M::CLI_NAME
            ));
            return Ok(());
        }
        let model_parent_dir =
            model_parent_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
        if epoch == 0 {
            crate::load_initial_model::load_initial_model(&model_parent_dir, M::API_NAME).await?;
        }

        log_info(format!(
            "Launching inference server for model {}",
            M::CLI_NAME,
        ));
        // let mut model_path = model_parent_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
        let model_path = format!("{}/model", model_parent_dir);
        log_info(format!("Using model folder path: {}", model_path));
        let (sglang_port, process) = launch_sglang_server_process::<M>(
            &model_path,
            self.num_gpus,
            use_tool,
            self.sglang_server_log_path.as_deref(),
        )
        .await?;
        log_info(format!(
            "SGLang server is listening on port {}",
            sglang_port
        ));

        self.inference_server_handle = Some(InferenceServerHandle {
            epoch,
            use_tool,
            sglang_port: Some(sglang_port),
            process: Some(process),
        });
        log_info("Inference server launched");
        Ok(())
    }

    async fn ensure_inference_server_shut_down<M: LlmModelMarker>(&mut self) {
        if let Some(handle) = self.inference_server_handle.take() {
            log_info(format!(
                "Shutting down inference server (stored_epoch={}, stored_use_tool={}, has_port={}, has_process={})...",
                handle.epoch,
                handle.use_tool,
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
            if model_uses_sglang::<M>() {
                log_info(
                    "ensure_inference_server_shut_down: no inference server handle present; checking for stale sglang process on configured port",
                );
                best_effort_shutdown_stale_sglang_server().await;
            } else {
                log_info(
                    "ensure_inference_server_shut_down: no inference server handle present; nothing to shut down",
                );
            }
        }
        assert!(
            self.inference_server_handle.is_none(),
            "Inference server should be shut down, but it's still Some"
        );
    }

    async fn validate_model<M: LlmModelMarker>(&self, epoch: usize) -> Result<(), String> {
        log_info("Start validating model.");
        // assert!(self.validation_rollout_config.split == DatasetSplit::Validation);

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
            max_python_processes: self.max_python_processes,
            total_epochs: self.num_total_epochs,
        };
        rollout_all::<M, Validation>(validation_rollout_program_config).await;
        log_info("Finished validating model.");
        Ok(())
    }

    async fn collect_training_rollout<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        log_info("Collecting training rollout");
        self.remove_epoch_training_checkpoints::<M>(epoch)?;
        let Some(sglang_server_handle) = &self.inference_server_handle else {
            panic!("Orchestrator did not launch the sglang server before generating training set");
        };
        let llm_cli_args = LlmCliArgs {
            sglang_port: sglang_server_handle.sglang_port.clone(),
        };
        // assert!(self.training_set_rollout_config.split == DatasetSplit::Training);
        let training_set_rollout_program_config = RolloutProgramConfig {
            config_nickname: self.config_nickname.clone(),
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            client: self.client.clone(),
            max_rollout_concurrency: self.max_rollout_concurrency,
            llm_cli_args,
            rollout_time_limit_secs: self.training_rollout_time_limit_secs,
            max_python_processes: self.max_python_processes,
            total_epochs: self.num_total_epochs,
        };
        rollout_all::<M, Training>(training_set_rollout_program_config).await;
        log_info("Finished collecting training rollout");
        Ok(())
    }

    fn remove_epoch_training_checkpoints<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        let checkpoint_parent_dir =
            model_checkpoint_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
        let metrics_path =
            model_metrics_path_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
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
        self.destroy_epoch_training_trajectories::<M>(epoch)?;
        self.cleanup_epoch_model_dir_if_not_best::<M>(epoch)?;
        Ok(())
    }

    fn destroy_epoch_checkpoint_folder<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        let checkpoint_parent_dir =
            model_checkpoint_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
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
        let training_logs = AssetFileDirectTreeActionLogs::<M, Training> {
            nickname: self.config_nickname.clone(),
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            _phantom: std::marker::PhantomData,
        };
        self.delete_file_if_exists(
            &training_logs.actions_file_path(),
            &format!("training rollout action logs for epoch {}", epoch),
        )?;

        let validation_logs = AssetFileDirectTreeActionLogs::<M, Validation> {
            nickname: self.config_nickname.clone(),
            rollout_config: self.validation_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            epoch,
            _phantom: std::marker::PhantomData,
        };
        self.delete_file_if_exists(
            &validation_logs.actions_file_path(),
            &format!("validation rollout action logs for epoch {}", epoch),
        )?;

        Ok(())
    }

    fn destroy_epoch_training_trajectories<M: LlmModelMarker>(
        &self,
        epoch: usize,
    ) -> Result<(), String> {
        let asset_file_training_trajectories = AssetFileTrainingTrajectories {
            config_nickname: self.config_nickname.clone(),
            epoch,
            rollout_config: self.training_set_rollout_config.clone(),
            posterior_calculation_config: self.posterior_calculation_config.clone(),
            cumulative_avg_abs_advantage_cutoff: self.cumulative_avg_abs_advantage_cutoff,
            advantage_calculation_policy: self.advantage_calculation_policy,
            _phantom: std::marker::PhantomData::<M>,
        };
        self.delete_file_if_exists(
            &asset_file_training_trajectories.file_path(),
            &format!("training trajectories sqlite for epoch {}", epoch),
        )
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
            model_parent_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
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
                model_parent_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
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
        let model_parent_dir =
            model_parent_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
        let checkpoints_parent_dir =
            model_checkpoint_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
        let final_model_output_parent_dir =
            model_parent_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch + 1)?;
        let training_summary_parent_dir =
            training_summary_parent_dir_from_template(M::CLI_NAME, &self.config_nickname, epoch)?;
        // first we need to write the training config to the expected location
        let training_config = PythonTrainingConfig {
            common: self.training_config_common.clone(),
            training_time: self.training_time,
            num_iterations_limit: self.num_iterations_limit,
            model_parent_dir,
            training_trajectory_sqlite_path,
            checkpoints_parent_dir,
            final_model_output_parent_dir,
            training_summary_parent_dir,
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
