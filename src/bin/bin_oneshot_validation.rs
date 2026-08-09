use std::backtrace::Backtrace;
use std::path::Path;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use ordered_float::NotNan;
use proctitle::set_title;
use serde::Deserialize;

use credit_assignment::{
    check_python_env::check_sympy_availability,
    chunked_judging::{
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT, DEFAULT_CACHE_VERSION,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL, judge_requests, read_judging_outputs,
    },
    constants,
    directories::{
        action_logs_oneshot_path, base_model_dir, inference_wrapper_log_path,
        oneshot_epochs_parent_dir, oneshot_model_parent_dir, text_logger_summary_path,
        text_logger_verbose_path, training_summary_oneshot_parent_dir, tree_artifacts_oneshot_path,
        tree_judgments_oneshot_path,
    },
    get_accuracy::get_accuracy_from_tree_judgments_at_path,
    hybrid_dataset::{DatasetSplit, Validation},
    json_toml_utils::read_json,
    launch_inference_wrapper::{
        self, InferenceBackend, best_effort_shutdown_stale_inference_wrapper,
        shut_down_inference_wrapper_process, update_inference_model,
    },
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    oneshot_training_summary::{
        derive_phase_log_path, ensure_parent_dir_exists, read_existing_validation_summary,
        read_oneshot_training_epoch_stats, write_training_summary,
    },
    oneshot_utils::oneshot_epoch_model_ready,
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    python_training_config::TrainingHyperparameters,
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::RolloutConfig,
    tree_artifact::{TreeJudgment, read_available_tree_artifact_chunks},
    tree_to_action::BranchingRuntimeOptions,
    utils::configure_mount_dir,
};
use reqwest::Client;
use research_utility::progress_text_logger::{
    ProgressTextLogger, log_info, log_key_value_pair, log_state, log_warning,
};

fn validation_max_concurrent_rollout(_num_gpus: usize) -> usize {
    200
}

fn tree_artifact_has_done_marker(tree_artifact_path: &str) -> bool {
    let path = Path::new(tree_artifact_path);
    if path.is_file() {
        return true;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            return false;
        };
        file_name.starts_with("chunk_") && file_name.ends_with("_done")
    })
}

fn requested_validation_epochs(num_oneshot_epochs: usize, epoch_interval: usize) -> Vec<usize> {
    assert!(epoch_interval > 0, "--epoch-interval must be positive");
    (0..=num_oneshot_epochs)
        .filter(|epoch| *epoch == 0 || *epoch % epoch_interval == 0)
        .collect()
}

async fn judge_validation_tree_artifacts<M: LlmModelMarker>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) {
    let artifacts = read_available_tree_artifact_chunks::<M, Validation>(tree_artifact_path)
        .unwrap_or_else(|err| panic!("failed to read marked validation tree chunks: {}", err));
    let requests = artifacts
        .iter()
        .flat_map(|artifact| artifact.to_judging_requests(DEFAULT_CACHE_VERSION))
        .collect::<Vec<_>>();
    let judging_output_jsonl_path = format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/validation_judging_outputs.jsonl"
    );
    let cache_dir =
        format!("{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/judgment_cache");
    let escalation_jsonl_path = format!(
        "{mount_dir}/small_files/{model_cli_name}/{config_nickname}/judgment_escalations.jsonl"
    );
    let summary = judge_requests(
        requests,
        std::path::Path::new(&judging_output_jsonl_path),
        std::path::Path::new(&cache_dir),
        std::path::Path::new(&escalation_jsonl_path),
        DEFAULT_CACHE_VERSION,
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL,
    )
    .await
    .unwrap_or_else(|err| panic!("failed to judge validation tree artifacts: {}", err));
    log_info(format!("Validation judging summary: {summary:#?}"));

    let outputs = read_judging_outputs(std::path::Path::new(&judging_output_jsonl_path))
        .unwrap_or_else(|err| panic!("failed to read validation judging outputs: {}", err));
    let mut outputs_by_artifact_id = std::collections::BTreeMap::<String, Vec<_>>::new();
    for output in outputs {
        let Some(artifact_id) = output.request.artifact_id.clone() else {
            continue;
        };
        outputs_by_artifact_id
            .entry(artifact_id)
            .or_default()
            .push(output);
    }
    if let Some(parent) = std::path::Path::new(tree_judgment_jsonl_path).parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "failed to create validation tree judgment parent {}: {}",
                parent.display(),
                err
            )
        });
    }
    let mut file = std::fs::File::create(tree_judgment_jsonl_path).unwrap_or_else(|err| {
        panic!(
            "failed to create validation tree judgment JSONL {}: {}",
            tree_judgment_jsonl_path, err
        )
    });
    for artifact in artifacts {
        let outputs = outputs_by_artifact_id
            .remove(&artifact.artifact_id)
            .unwrap_or_default();
        let judgment = TreeJudgment::from_judging_outputs(
            artifact.artifact_id.clone(),
            DEFAULT_CACHE_VERSION.to_string(),
            Validation::dataset_file_postfix(),
            artifact.question.flat_id.0,
            outputs,
        )
        .unwrap_or_else(|err| panic!("failed to build validation tree judgment: {}", err));
        serde_json::to_writer(&mut file, &judgment)
            .unwrap_or_else(|err| panic!("failed to serialize validation tree judgment: {}", err));
        use std::io::Write;
        writeln!(file)
            .unwrap_or_else(|err| panic!("failed to write validation tree judgment: {}", err));
    }
}

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(short = 'c', long)]
    config_path: String,
    #[arg(long, default_value = "all")]
    phase: ValidationPhase,
    #[arg(long, default_value_t = 1)]
    epoch_interval: usize,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ValidationPhase {
    All,
    Rollout,
    Judge,
    Score,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Args {
    model_cli_name: String,
    config_nickname_training: String,
    config_nickname_generation: String,
    use_tool: bool,
    num_oneshot_epochs: usize,
    #[serde(default)]
    validation_total_epochs: Option<usize>,
    validation_rollout_secs: usize,
    training_hyperparameters: TrainingHyperparameters,
    oneshot_per_epoch_training_time: f32,
    num_iterations_limit: usize,
    num_gpus: usize,
    inference_backend: InferenceBackend,
    training_trajectory_len_cutoff: usize,
    #[serde(default)]
    training_set_sort_mode: String,
    #[serde(default)]
    total_time_limit_hours: f32,
    mount_dir: String,
    generation_mount_dir: String,
}

async fn run_oneshot_validation<M: LlmModelMarker>(
    model_cli_name: &str,
    config_nickname_training: &str,
    generation_mount_dir: &str,
    mount_dir: &str,
    validation_rollout_config: RolloutConfig<Validation>,
    posterior_calculation_config: PosteriorCalculationConfig,
    num_oneshot_epochs: usize,
    validation_rollout_secs: usize,
    num_gpus: usize,
    inference_backend: InferenceBackend,
    inference_wrapper_log_path: &str,
    use_tool: bool,
    phase: ValidationPhase,
    epoch_interval: usize,
) {
    let client = Client::new();
    let oneshot_training_summary_parent_dir =
        training_summary_oneshot_parent_dir(mount_dir, model_cli_name, config_nickname_training);
    let oneshot_model_output_root =
        oneshot_epochs_parent_dir(mount_dir, model_cli_name, config_nickname_training);
    let (already_validated_epochs, mut validation_accuracies) =
        read_existing_validation_summary(&oneshot_training_summary_parent_dir);
    let requested_epochs = requested_validation_epochs(num_oneshot_epochs, epoch_interval);

    if matches!(phase, ValidationPhase::All | ValidationPhase::Score)
        && requested_epochs
            .iter()
            .all(|epoch| already_validated_epochs.contains(epoch))
    {
        log_info("All requested oneshot epochs are already validated; skipping validation phase");
        return;
    }

    log_info(format!(
        "Will validate epochs {:?} from base epoch plus trained epochs 1..={} with epoch interval {} (already validated: {:?})",
        requested_epochs, num_oneshot_epochs, epoch_interval, already_validated_epochs
    ));
    let mut launched_inference = None;
    if matches!(phase, ValidationPhase::All | ValidationPhase::Rollout) {
        best_effort_shutdown_stale_inference_wrapper().await;

        let launch_model_parent_dir = base_model_dir(mount_dir, model_cli_name);
        let launch_model_path = format!("{}/model", launch_model_parent_dir);
        let (sglang_port, handle) = launch_inference_wrapper::launch_inference_wrapper_process(
            inference_backend,
            &launch_model_path,
            model_cli_name,
            config_nickname_training,
            0,
            M::API_NAME,
            num_gpus,
            inference_wrapper_log_path,
        )
        .await
        .unwrap_or_else(|err| panic!("failed to launch inference server: {}", err));
        log_info(format!(
            "Inference server listening on port {} for validation",
            sglang_port
        ));
        launched_inference = Some((sglang_port, handle));
    }

    for epoch in requested_epochs {
        if matches!(phase, ValidationPhase::All | ValidationPhase::Score)
            && already_validated_epochs.contains(&epoch)
        {
            log_info(format!("Epoch {}: already validated; skipping", epoch));
            continue;
        }
        log_state(format!(
            "One-shot validation for epoch {}/{}",
            epoch, num_oneshot_epochs
        ));

        if matches!(phase, ValidationPhase::All | ValidationPhase::Rollout) && epoch > 0 {
            let wait_deadline = std::time::Instant::now() + Duration::from_secs(600);
            while !oneshot_epoch_model_ready(&oneshot_model_output_root, epoch) {
                if std::time::Instant::now() >= wait_deadline {
                    log_warning(format!(
                        "Epoch {} model was not ready after 600s; exiting validation early so a later run can resume",
                        epoch
                    ));
                    break;
                }
                log_info(format!(
                    "Epoch {} model is not ready yet; polling again in 30s",
                    epoch
                ));
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
            if !oneshot_epoch_model_ready(&oneshot_model_output_root, epoch) {
                break;
            }
            let model_path = format!(
                "{}/model",
                oneshot_model_parent_dir(
                    mount_dir,
                    model_cli_name,
                    config_nickname_training,
                    epoch,
                )
            );
            if !Path::new(&model_path).exists() {
                log_warning(format!(
                    "Skipping validation for epoch {} because model path is missing: {}",
                    epoch, model_path
                ));
                continue;
            }
            log_info(format!(
                "Epoch {}: Updating inference model weights to {}",
                epoch, model_path
            ));
            let (sglang_port, _) = launched_inference
                .as_ref()
                .expect("validation rollout phase must have inference server");
            update_inference_model(*sglang_port, &model_path, inference_wrapper_log_path)
                .await
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to update inference model to epoch {}: {}",
                        epoch, err
                    )
                });
        }

        let validation_action_log_path = action_logs_oneshot_path::<Validation>(
            generation_mount_dir,
            model_cli_name,
            config_nickname_training,
            epoch,
        )
        .unwrap_or_else(|err| {
            panic!(
                "failed to resolve one-shot validation action logs path for epoch {}: {}",
                epoch, err
            )
        });
        let validation_tree_artifact_path = tree_artifacts_oneshot_path::<Validation>(
            mount_dir,
            model_cli_name,
            config_nickname_training,
            epoch,
        )
        .unwrap_or_else(|err| {
            panic!(
                "failed to resolve one-shot validation tree artifacts path for epoch {}: {}",
                epoch, err
            )
        });
        let validation_tree_judgment_path = tree_judgments_oneshot_path::<Validation>(
            mount_dir,
            model_cli_name,
            config_nickname_training,
            epoch,
        )
        .unwrap_or_else(|err| {
            panic!(
                "failed to resolve one-shot validation tree judgments path for epoch {}: {}",
                epoch, err
            )
        });
        if matches!(phase, ValidationPhase::All | ValidationPhase::Rollout) {
            let (sglang_port, _) = launched_inference
                .as_ref()
                .expect("validation rollout phase must have inference server");
            let validation_program_config = RolloutProgramConfig {
                config_nickname: config_nickname_training.to_string(),
                rollout_config: validation_rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
                epoch,
                client: client.clone(),
                inference_endpoint: InferenceEndpoint::SglangPort(*sglang_port),
                rollout_secs: validation_rollout_secs,
                finish_all_questions: true,
                total_epochs: num_oneshot_epochs,
                action_log_store_override_path: Some(validation_action_log_path.clone()),
                use_tool,
                fixed_temperature: NotNan::new(constants::VALIDATION_TEMPERATURE).unwrap(),
                max_concurrent_rollout: validation_max_concurrent_rollout(num_gpus),
                branching_options: BranchingRuntimeOptions::default(),
                tree_artifact_output_path: Some(validation_tree_artifact_path.clone()),
                tree_artifact_chunk_question_count: None,
                question_flat_id_start: None,
                question_flat_id_end: None,
                question_flat_ids: None,
            };
            let validation_summary =
                rollout_all::<M, Validation>(mount_dir, validation_program_config).await;
            log_info(format!(
                "Epoch {}: Validation rollout finished ({:.3}s, {} LLM calls)",
                epoch, validation_summary.elapsed_secs, validation_summary.total_llm_calls,
            ));
        }

        if matches!(phase, ValidationPhase::All | ValidationPhase::Judge) {
            if !tree_artifact_has_done_marker(&validation_tree_artifact_path) {
                log_warning(format!(
                    "Epoch {}: validation tree artifacts are incomplete or missing at {}; skipping judge phase for this epoch",
                    epoch, validation_tree_artifact_path
                ));
                continue;
            }
            judge_validation_tree_artifacts::<M>(
                &validation_tree_artifact_path,
                &validation_tree_judgment_path,
                mount_dir,
                model_cli_name,
                config_nickname_training,
                epoch,
            )
            .await;
        }

        if matches!(phase, ValidationPhase::All | ValidationPhase::Score) {
            if !tree_artifact_has_done_marker(&validation_tree_artifact_path) {
                log_warning(format!(
                    "Epoch {}: validation tree artifacts are incomplete or missing at {}; skipping score phase for this epoch",
                    epoch, validation_tree_artifact_path
                ));
                continue;
            }
            if !Path::new(&validation_tree_judgment_path).exists() {
                log_warning(format!(
                    "Epoch {}: validation tree judgments are missing at {}; skipping score phase for this epoch",
                    epoch, validation_tree_judgment_path
                ));
                continue;
            }
            let accuracy_stats = get_accuracy_from_tree_judgments_at_path::<M, Validation>(
                &validation_tree_artifact_path,
                &validation_tree_judgment_path,
                "Validation accuracy (one-shot)",
            )
            .await;
            if let Some(accuracies) = accuracy_stats.accuracy_tuple() {
                validation_accuracies.insert(epoch, accuracies);
                log_key_value_pair(
                    format!("epoch_{}_validation_accuracy_deepmath", epoch),
                    accuracies.1.to_string(),
                );
                log_key_value_pair(
                    format!("epoch_{}_validation_accuracy_math", epoch),
                    accuracies.2.to_string(),
                );
                log_info(format!(
                    "Epoch {}: Validation accuracy (avg={:.6}, deepmath={:.6}, math={:.6}, numinamath={:.6})",
                    epoch, accuracies.0, accuracies.1, accuracies.2, accuracies.3,
                ));
            } else {
                log_warning(format!(
                    "Epoch {}: No validation accuracy data available",
                    epoch
                ));
            }
        }

        if Path::new(&validation_action_log_path).exists() {
            match std::fs::remove_dir_all(&validation_action_log_path) {
                Ok(()) => log_info(format!(
                    "Epoch {}: Cleaned up validation action logs at {}",
                    epoch, validation_action_log_path
                )),
                Err(err) => log_warning(format!(
                    "Epoch {}: Failed to clean up validation action logs at {}: {}",
                    epoch, validation_action_log_path, err
                )),
            }
        }

        if matches!(phase, ValidationPhase::All | ValidationPhase::Score) {
            let training_epoch_stats =
                read_oneshot_training_epoch_stats(&oneshot_model_output_root, num_oneshot_epochs);
            write_training_summary(
                &oneshot_training_summary_parent_dir,
                epoch,
                num_oneshot_epochs,
                &validation_accuracies,
                &training_epoch_stats.throughputs,
                &training_epoch_stats.samples_trained,
                &training_epoch_stats.iterations_trained_cumulative,
                &training_epoch_stats.longest_non_oom_trajectory_lengths,
            );
        }
    }

    if let Some((_, mut handle)) = launched_inference {
        log_info("Shutting down inference server after oneshot validation rollout");
        let _ = handle.stop_signal_tx.send(true);
        shut_down_inference_wrapper_process(&mut handle.child).await;
        let _ = handle.listener_handle.await;
        log_info("Inference server shut down");
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic occurred: {}", info);
        let rust_backtrace = std::env::var("RUST_BACKTRACE").ok();
        if matches!(rust_backtrace.as_deref(), Some("1") | Some("full")) {
            let backtrace = Backtrace::force_capture();
            eprintln!("backtrace:\n{}", backtrace);
        }
        std::process::abort();
    }));
    dotenvy::dotenv().ok();
    let CliArgs {
        config_path,
        phase,
        epoch_interval,
        login_smoke,
    } = CliArgs::parse();
    let config_contents = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read config file '{}': {}", config_path, err));
    let Args {
        model_cli_name,
        config_nickname_training,
        use_tool,
        num_oneshot_epochs,
        validation_total_epochs,
        validation_rollout_secs,
        num_gpus,
        inference_backend,
        mount_dir,
        generation_mount_dir,
        ..
    } = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));
    let validation_total_epochs = validation_total_epochs.unwrap_or(num_oneshot_epochs);

    let process_title = format!(
        "oneshot_validation_{}_{}",
        model_cli_name, config_nickname_training
    );
    set_title(&process_title);
    check_sympy_availability().unwrap();
    assert!(num_gpus > 0, "--num-gpus must be positive");
    assert!(
        validation_total_epochs > 0,
        "validation_total_epochs/num_oneshot_epochs must be positive"
    );
    assert!(epoch_interval > 0, "--epoch-interval must be positive");
    let validation_rollout_config: RolloutConfig<Validation> =
        read_json(credit_assignment::directories::VALIDATION_ROLLOUT_CONFIG_PATH).unwrap();
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )
    .unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let model_name = LlmModelName::from_str(&model_cli_name, true).unwrap();
    if login_smoke {
        println!(
            "login-smoke passed for bin_oneshot_validation: model={}, training_config={}, phase={:?}, validation_epochs={}, epoch_interval={}, requested_epochs={:?}, backend={:?}, num_gpus={}",
            model_cli_name,
            config_nickname_training,
            phase,
            validation_total_epochs,
            epoch_interval,
            requested_validation_epochs(validation_total_epochs, epoch_interval),
            inference_backend,
            num_gpus
        );
        return;
    }
    configure_mount_dir(&mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));

    let inference_wrapper_log_path = derive_phase_log_path(
        &inference_wrapper_log_path(&mount_dir, &model_cli_name, &config_nickname_training),
        "validation",
    );
    let text_log_summary_path = derive_phase_log_path(
        &text_logger_summary_path(&mount_dir, &model_cli_name, &config_nickname_training),
        "validation",
    );
    let text_log_verbose_path = derive_phase_log_path(
        &text_logger_verbose_path(&mount_dir, &model_cli_name, &config_nickname_training),
        "validation",
    );

    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare inference wrapper log directory: {}", err));
    ensure_parent_dir_exists(&text_log_summary_path).unwrap_or_else(|err| {
        panic!(
            "failed to prepare validation summary log directory: {}",
            err
        )
    });
    ensure_parent_dir_exists(&text_log_verbose_path).unwrap_or_else(|err| {
        panic!(
            "failed to prepare validation verbose log directory: {}",
            err
        )
    });
    ProgressTextLogger::initialize(text_log_summary_path, text_log_verbose_path)
        .await
        .unwrap();

    match model_name {
        LlmModelName::Gemma3_4b => {
            run_oneshot_validation::<Gemma3_4BIt>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
        LlmModelName::Llama31_8b => {
            run_oneshot_validation::<Llama31_8BInstruct>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
        LlmModelName::Mistral7bInstructV03 => {
            run_oneshot_validation::<Mistral7BInstructV03>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
        LlmModelName::Qwen3_06b => {
            run_oneshot_validation::<Qwen3_06B>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
        LlmModelName::Qwen3_4b => {
            run_oneshot_validation::<Qwen3_4B>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
        LlmModelName::Qwen25_7b => {
            run_oneshot_validation::<Qwen25_7B>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
        LlmModelName::Qwen35_08b => {
            run_oneshot_validation::<Qwen35_08B>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
        LlmModelName::Qwen35_4b => {
            run_oneshot_validation::<Qwen35_4B>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                validation_total_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
                phase,
                epoch_interval,
            )
            .await
        }
    }

    ProgressTextLogger::shutdown().await.unwrap();
}
