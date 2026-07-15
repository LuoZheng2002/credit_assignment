use std::backtrace::Backtrace;
use std::collections::BTreeMap;
use std::path::Path;

use clap::{Parser, ValueEnum};
use ordered_float::NotNan;
use proctitle::set_title;
use serde::Deserialize;
use tokio::time::Instant;

use credit_assignment::{
    check_python_env::check_sympy_availability,
    constants,
    constants::get_max_concurrent_rollout,
    directories::{
        action_logs_oneshot_path, base_model_dir, inference_wrapper_log_path,
        oneshot_epochs_parent_dir, oneshot_model_parent_dir, text_logger_summary_path,
        text_logger_verbose_path, training_summary_oneshot_parent_dir,
        training_trajectories_oneshot_path, training_wrapper_log_path,
    },
    get_accuracy::get_accuracy_at_path,
    hybrid_dataset::Validation,
    json_toml_utils::read_json,
    launch_inference_wrapper::{self, shut_down_inference_wrapper_process, update_inference_model},
    launch_training_wrapper::run_training_wrapper_and_wait,
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    oneshot_utils::{
        all_expected_oneshot_model_outputs_exist, detect_oneshot_artifacts,
        oneshot_aggregated_summary_path, read_oneshot_epoch_count_from_summary,
        read_oneshot_run_manifest, write_oneshot_run_manifest,
    },
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    python_training_config::{PythonTrainingConfig, TrainingHyperparameters, TrainingMode},
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::RolloutConfig,
    utils::configure_mount_dir,
};
use reqwest::Client;
use research_utility::progress_text_logger::{
    ProgressTextLogger, log_info, log_key_value_pair, log_state, log_warning,
};

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(short = 'c', long)]
    config_path: String,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Args {
    model_cli_name: String,
    config_nickname_training: String,
    config_nickname_rollout: String,
    use_tool: bool,
    num_oneshot_epochs: usize,
    validation_rollout_secs: usize,
    training_hyperparameters: TrainingHyperparameters,
    oneshot_per_epoch_training_time: f32,
    num_iterations_limit: usize,
    num_gpus: usize,
    training_trajectory_len_cutoff: usize,
    #[serde(default)]
    total_time_limit_hours: f32,
    mount_dir: String,
    rollout_mount_dir: String,
}

fn ensure_parent_dir_exists(file_path: &str) -> Result<(), String> {
    let Some(parent) = Path::new(file_path).parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(parent).map_err(|err| {
        format!(
            "Failed to create parent directory {}: {}",
            parent.display(),
            err
        )
    })
}

fn write_training_summary(
    summary_parent_dir: &str,
    latest_epoch: usize,
    num_oneshot_epochs: usize,
    validation_accuracies: &BTreeMap<usize, (f32, f32, f32, f32)>,
    training_throughputs: &BTreeMap<usize, f32>,
) {
    std::fs::create_dir_all(summary_parent_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create training summary parent dir {}: {}",
            summary_parent_dir, err
        )
    });

    let mut accuracies_json: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (epoch, (avg, deepmath, math, _numinamath)) in validation_accuracies.iter() {
        accuracies_json.insert(
            format!("epoch_{}", epoch),
            serde_json::json!({
                "avg": avg,
                "deepmath": deepmath,
                "math": math,
            }),
        );
    }

    let mut throughputs_json: BTreeMap<String, f32> = BTreeMap::new();
    for (epoch, throughput) in training_throughputs.iter() {
        throughputs_json.insert(format!("epoch_{}", epoch), *throughput);
    }

    let latest_epoch_data = {
        let (avg, deepmath, math, _numinamath) = validation_accuracies
            .get(&latest_epoch)
            .map(|(a, b, c, d)| (*a, *b, *c, *d))
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        let throughput = training_throughputs
            .get(&latest_epoch)
            .copied()
            .unwrap_or(0.0);
        serde_json::json!({
            "epoch": latest_epoch,
            "validation_accuracy": {
                "avg": avg,
                "deepmath": deepmath,
                "math": math,
            },
            "training_throughput": throughput,
        })
    };

    // 1. Per-epoch snapshot: oneshot_per_epoch_summary_epoch_{N}.json
    let epoch_output_path = Path::new(summary_parent_dir).join(format!(
        "oneshot_per_epoch_summary_epoch_{}.json",
        latest_epoch
    ));
    std::fs::write(
        &epoch_output_path,
        serde_json::to_string_pretty(&latest_epoch_data).unwrap() + "\n",
    )
    .unwrap_or_else(|err| {
        panic!(
            "Failed to write per-epoch oneshot summary to {}: {}",
            epoch_output_path.display(),
            err
        )
    });
    log_info(format!(
        "Wrote per-epoch oneshot summary to {}",
        epoch_output_path.display()
    ));

    // 2. Aggregated file: oneshot_aggregated_summary.json (all epochs so far)
    let accumulated_path = oneshot_aggregated_summary_path(summary_parent_dir);
    let payload = serde_json::json!({
        "latest_epoch": latest_epoch,
        "num_oneshot_epochs": num_oneshot_epochs,
        "validation_accuracies": accuracies_json,
        "training_throughputs": throughputs_json,
    });
    std::fs::write(
        &accumulated_path,
        serde_json::to_string_pretty(&payload).unwrap() + "\n",
    )
    .unwrap_or_else(|err| {
        panic!(
            "Failed to write aggregated oneshot summary to {}: {}",
            accumulated_path.display(),
            err
        )
    });
    log_info(format!(
        "Wrote aggregated oneshot summary to {}",
        accumulated_path.display()
    ));
}

async fn run_oneshot_training<M: LlmModelMarker>(
    model_cli_name: &str,
    config_nickname_training: &str,
    config_nickname_rollout: &str,
    rollout_mount_dir: &str,
    mount_dir: &str,
    validation_rollout_config: RolloutConfig<Validation>,
    posterior_calculation_config: PosteriorCalculationConfig,
    num_oneshot_epochs: usize,
    validation_rollout_secs: usize,
    training_hyperparameters: TrainingHyperparameters,
    oneshot_per_epoch_training_time: f32,
    num_iterations_limit: usize,
    num_gpus: usize,
    training_trajectory_len_cutoff: usize,
    inference_wrapper_log_path: &str,
    training_wrapper_log_path: &str,
    use_tool: bool,
) {
    let client = Client::new();

    // Resolve one-shot paths (use rollout mount dir to find trajectories on the rollout volume)
    let oneshot_trajectories_dir = training_trajectories_oneshot_path(
        rollout_mount_dir,
        model_cli_name,
        config_nickname_rollout,
    );
    let oneshot_trajectories_msgpack_path = Path::new(&oneshot_trajectories_dir)
        .join("trajectories.msgpack")
        .to_string_lossy()
        .into_owned();

    // Training trajectories must be pre-generated by the oneshot rollout binary
    if !Path::new(&oneshot_trajectories_msgpack_path).exists() {
        panic!(
            "Training trajectories file not found at '{}'. \
             Run the oneshot rollout binary (bin_oneshot_rollout) first to generate it.",
            oneshot_trajectories_msgpack_path
        );
    }
    log_info(format!(
        "Using pre-generated training trajectories at {}",
        oneshot_trajectories_msgpack_path
    ));

    let oneshot_training_summary_parent_dir =
        training_summary_oneshot_parent_dir(mount_dir, model_cli_name, config_nickname_training);
    // ================================================================
    // Phase 1: Train all oneshot epochs in a single process
    //        (optimizer/data-cursor continuity is kept in memory only;
    //         no training-state checkpoint resume is supported)
    // ================================================================

    let oneshot_model_output_root =
        oneshot_epochs_parent_dir(mount_dir, model_cli_name, config_nickname_training);

    let base_model_parent_dir = base_model_dir(mount_dir, model_cli_name);

    let mut training_throughputs: BTreeMap<usize, f32> = BTreeMap::new();
    for epoch in 0..num_oneshot_epochs {
        training_throughputs.insert(epoch, 0.0);
    }

    let previous_artifacts_detected = detect_oneshot_artifacts(
        &oneshot_training_summary_parent_dir,
        &oneshot_model_output_root,
    );
    if previous_artifacts_detected {
        let previous_num_oneshot_epochs =
            read_oneshot_run_manifest(&oneshot_training_summary_parent_dir)
                .map(|manifest| manifest.num_oneshot_epochs)
                .or_else(|| {
                    read_oneshot_epoch_count_from_summary(&oneshot_training_summary_parent_dir)
                });

        let Some(previous_num_oneshot_epochs) = previous_num_oneshot_epochs else {
            log_warning(format!(
                "Detected prior oneshot artifacts under {} but could not determine the previous num_oneshot_epochs; exiting early because oneshot training resume is disabled. Clean up the old artifacts to start fresh.",
                oneshot_model_output_root
            ));
            return;
        };

        if previous_num_oneshot_epochs != num_oneshot_epochs {
            log_warning(format!(
                "Detected prior oneshot artifacts with num_oneshot_epochs={} but current request uses {}; exiting early because oneshot training resume is disabled.",
                previous_num_oneshot_epochs, num_oneshot_epochs
            ));
            return;
        }

        if !all_expected_oneshot_model_outputs_exist(&oneshot_model_output_root, num_oneshot_epochs)
        {
            log_warning(format!(
                "Detected prior oneshot artifacts for num_oneshot_epochs={} but not all expected model outputs are present under {}; exiting early because oneshot training resume is disabled.",
                num_oneshot_epochs, oneshot_model_output_root
            ));
            return;
        }

        log_info(format!(
            "Detected matching prior oneshot artifacts for num_oneshot_epochs={}; skipping training and continuing with remaining validation epochs",
            num_oneshot_epochs
        ));
    } else {
        write_oneshot_run_manifest(&oneshot_training_summary_parent_dir, num_oneshot_epochs);
        log_state(format!(
            "Starting oneshot training for {} epoch(s) in a single process",
            num_oneshot_epochs
        ));

        let training_config = PythonTrainingConfig {
            hyperparameters: training_hyperparameters.clone(),
            num_iterations_limit,
            training_trajectory_len_cutoff,
            model_cli_name: model_cli_name.to_string(),
            config_nickname: config_nickname_training.to_string(),
            training_mode: TrainingMode::OneShot {
                per_epoch_training_time: oneshot_per_epoch_training_time,
                num_oneshot_epochs,
                model_output_root: oneshot_model_output_root.clone(),
                training_summary_dir: oneshot_model_output_root.clone(),
                base_model_parent_dir,
            },
        };

        let training_start_time = Instant::now();
        run_training_wrapper_and_wait(
            num_gpus,
            M::API_NAME,
            &training_config,
            &oneshot_trajectories_msgpack_path,
            training_wrapper_log_path,
        )
        .await
        .unwrap_or_else(|err| panic!("Oneshot training failed: {}", err));
        let elapsed_secs = training_start_time.elapsed().as_secs_f32();
        log_info(format!(
            "Oneshot training of {} epoch(s) finished in {:.3}s",
            num_oneshot_epochs, elapsed_secs
        ));
    }

    log_state("All oneshot training epochs completed; starting validation phase");

    // ================================================================
    // Phase 2: Validate all models (base model + all trained epochs)
    // ================================================================

    // Detect already-validated epochs by reading the aggregated summary.
    let already_validated_epochs: std::collections::HashSet<usize> = {
        let aggregated_path = oneshot_aggregated_summary_path(&oneshot_training_summary_parent_dir);
        if aggregated_path.exists() {
            std::fs::read_to_string(&aggregated_path)
                .ok()
                .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
                .and_then(|parsed| {
                    parsed
                        .get("validation_accuracies")
                        .and_then(|v| v.as_object())
                        .map(|obj| {
                            obj.keys()
                                .filter_map(|k| k.strip_prefix("epoch_")?.parse::<usize>().ok())
                                .collect()
                        })
                })
                .unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        }
    };

    let mut validation_accuracies: BTreeMap<usize, (f32, f32, f32, f32)> = BTreeMap::new();

    // Pre-populate validation accuracies from summary (for already-validated epochs)
    if !already_validated_epochs.is_empty() {
        let aggregated_path = oneshot_aggregated_summary_path(&oneshot_training_summary_parent_dir);
        if let Ok(content) = std::fs::read_to_string(&aggregated_path) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(acc_map) = parsed
                    .get("validation_accuracies")
                    .and_then(|v| v.as_object())
                {
                    for (key, value) in acc_map {
                        if let Some(epoch_str) = key.strip_prefix("epoch_") {
                            if let Ok(epoch) = epoch_str.parse::<usize>() {
                                let avg =
                                    value.get("avg").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                                let deepmath = value
                                    .get("deepmath")
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(0.0)
                                    as f32;
                                let math = value.get("math").and_then(|v| v.as_f64()).unwrap_or(0.0)
                                    as f32;
                                validation_accuracies.insert(epoch, (avg, deepmath, math, 0.0));
                            }
                        }
                    }
                }
            }
        }
    }

    let epochs_to_validate: Vec<usize> = (0..=num_oneshot_epochs)
        .filter(|e| !already_validated_epochs.contains(e))
        .collect();

    if epochs_to_validate.is_empty() {
        log_info("All epochs already validated; skipping validation phase");
    } else {
        // Launch inference server ONCE with epoch 0 weights (base model)
        let launch_model_parent_dir = base_model_dir(mount_dir, model_cli_name);
        let launch_model_path = format!("{}/model", launch_model_parent_dir);

        log_info(format!(
            "Launching inference server once for all validation epochs (initial model: {})",
            launch_model_path
        ));
        let (sglang_port, mut handle) = launch_inference_wrapper::launch_inference_wrapper_process(
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
            "Inference server listening on port {} for all validation epochs",
            sglang_port
        ));

        for &epoch in &epochs_to_validate {
            log_state(format!(
                "One-shot validation for epoch {}/{}",
                epoch, num_oneshot_epochs
            ));

            if epoch > 0 {
                let model_path = format!(
                    "{}/model",
                    oneshot_model_parent_dir(
                        mount_dir,
                        model_cli_name,
                        config_nickname_training,
                        epoch,
                    )
                );
                log_info(format!(
                    "Epoch {}: Updating inference model weights to {}",
                    epoch, model_path
                ));
                update_inference_model(sglang_port, &model_path, inference_wrapper_log_path)
                    .await
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to update inference model to epoch {}: {}",
                            epoch, err
                        )
                    });
            }

            let validation_action_log_path = action_logs_oneshot_path::<Validation>(
                rollout_mount_dir,
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
            let validation_program_config = RolloutProgramConfig {
                config_nickname: config_nickname_training.to_string(),
                rollout_config: validation_rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
                epoch,
                client: client.clone(),
                inference_endpoint: InferenceEndpoint::SglangPort(sglang_port),
                rollout_secs: validation_rollout_secs,
                total_epochs: num_oneshot_epochs,
                action_log_store_override_path: Some(validation_action_log_path.clone()),
                use_tool,
                fixed_temperature: NotNan::new(constants::VALIDATION_TEMPERATURE).unwrap(),
                max_concurrent_rollout: get_max_concurrent_rollout(num_gpus),
            };
            let validation_summary =
                rollout_all::<M, Validation>(mount_dir, validation_program_config).await;
            log_info(format!(
                "Epoch {}: Validation rollout finished ({:.3}s, {} LLM calls)",
                epoch, validation_summary.elapsed_secs, validation_summary.total_llm_calls,
            ));

            // Read validation accuracy from one-shot paths
            let accuracy_stats = get_accuracy_at_path::<M, Validation>(
                &validation_action_log_path,
                validation_rollout_config.clone(),
                posterior_calculation_config.clone(),
                "Validation accuracy (one-shot)",
                use_tool,
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

            // Clean up validation action logs after reading accuracy
            let validation_action_log_path_cleanup = validation_action_log_path.clone();
            if Path::new(&validation_action_log_path_cleanup).exists() {
                match std::fs::remove_dir_all(&validation_action_log_path_cleanup) {
                    Ok(()) => log_info(format!(
                        "Epoch {}: Cleaned up validation action logs at {}",
                        epoch, validation_action_log_path_cleanup
                    )),
                    Err(err) => log_warning(format!(
                        "Epoch {}: Failed to clean up validation action logs at {}: {}",
                        epoch, validation_action_log_path_cleanup, err
                    )),
                }
            }

            // Write training summary (accumulated so far, one-shot path)
            write_training_summary(
                &oneshot_training_summary_parent_dir,
                epoch,
                num_oneshot_epochs,
                &validation_accuracies,
                &training_throughputs,
            );
        }

        // Shut down inference server once after all validation epochs
        log_info("Shutting down inference server after all validation epochs");
        let _ = handle.stop_signal_tx.send(true);
        shut_down_inference_wrapper_process(&mut handle.child).await;
        let _ = handle.listener_handle.await;
        log_info("Inference server shut down");
    }

    log_state("One-shot training completed for all epochs");
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
    let CliArgs { config_path } = CliArgs::parse();
    let config_contents = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read config file '{}': {}", config_path, err));
    let Args {
        model_cli_name,
        config_nickname_training,
        config_nickname_rollout,
        use_tool,
        num_oneshot_epochs,
        validation_rollout_secs,
        training_hyperparameters,
        oneshot_per_epoch_training_time,
        num_iterations_limit,
        num_gpus,
        training_trajectory_len_cutoff,
        mount_dir,
        rollout_mount_dir,
        ..
    } = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));
    let process_title = format!(
        "oneshot_training_{}_{}",
        model_cli_name, config_nickname_training
    );
    set_title(&process_title);
    check_sympy_availability().unwrap();
    configure_mount_dir(&mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));
    let inference_wrapper_log_path =
        inference_wrapper_log_path(&mount_dir, &model_cli_name, &config_nickname_training);
    let training_wrapper_log_path =
        training_wrapper_log_path(&mount_dir, &model_cli_name, &config_nickname_training);
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare inference wrapper log directory: {}", err));
    ensure_parent_dir_exists(&training_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare training wrapper log directory: {}", err));
    assert!(num_gpus > 0, "--num-gpus must be positive");
    assert!(
        num_oneshot_epochs > 0,
        "--num-oneshot-epochs must be positive"
    );
    log_info(format!("One-shot training will use num_gpus={}", num_gpus));

    let text_log_summary_path =
        text_logger_summary_path(&mount_dir, &model_cli_name, &config_nickname_training);
    let text_log_verbose_path =
        text_logger_verbose_path(&mount_dir, &model_cli_name, &config_nickname_training);
    ProgressTextLogger::initialize(text_log_summary_path, text_log_verbose_path)
        .await
        .unwrap();
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

    match model_name {
        LlmModelName::Gemma3_4b => {
            run_oneshot_training::<Gemma3_4BIt>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
        LlmModelName::Llama31_8b => {
            run_oneshot_training::<Llama31_8BInstruct>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
        LlmModelName::Mistral7bInstructV03 => {
            run_oneshot_training::<Mistral7BInstructV03>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
        LlmModelName::Qwen3_06b => {
            run_oneshot_training::<Qwen3_06B>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
        LlmModelName::Qwen3_4b => {
            run_oneshot_training::<Qwen3_4B>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
        LlmModelName::Qwen25_7b => {
            run_oneshot_training::<Qwen25_7B>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
        LlmModelName::Qwen35_08b => {
            run_oneshot_training::<Qwen35_08B>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
        LlmModelName::Qwen35_4b => {
            run_oneshot_training::<Qwen35_4B>(
                &model_cli_name,
                &config_nickname_training,
                &config_nickname_rollout,
                &rollout_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                training_hyperparameters,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                training_trajectory_len_cutoff,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                use_tool,
            )
            .await
        }
    }

    ProgressTextLogger::shutdown().await.unwrap();
}
