use std::backtrace::Backtrace;
use std::path::Path;

use clap::{Parser, ValueEnum};
use ordered_float::NotNan;
use proctitle::set_title;
use serde::Deserialize;

use credit_assignment::{
    check_python_env::check_sympy_availability,
    constants,
    constants::get_max_concurrent_rollout,
    directories::{
        action_logs_oneshot_path, base_model_dir, inference_wrapper_log_path,
        oneshot_epochs_parent_dir, oneshot_model_parent_dir, text_logger_summary_path,
        text_logger_verbose_path, training_summary_oneshot_parent_dir,
    },
    get_accuracy::get_accuracy_at_path,
    hybrid_dataset::Validation,
    json_toml_utils::read_json,
    launch_inference_wrapper::{
        self, InferenceBackend, best_effort_shutdown_stale_inference_wrapper,
        shut_down_inference_wrapper_process, update_inference_model,
    },
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen25_7B, Qwen3_06B, Qwen3_4B, Qwen35_08B, Qwen35_4B,
    },
    oneshot_training_summary::{
        derive_phase_log_path, detect_trained_oneshot_epochs, ensure_parent_dir_exists,
        prune_non_best_oneshot_models, read_existing_validation_summary,
        read_oneshot_training_epoch_stats, write_training_summary,
    },
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    python_training_config::TrainingHyperparameters,
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::RolloutConfig,
    utils::configure_mount_dir,
};
use reqwest::Client;
use research_utility::progress_text_logger::{
    ProgressTextLogger, log_info, log_key_value_pair, log_state, log_warning,
};

fn validation_max_concurrent_rollout(num_gpus: usize) -> usize {
    std::env::var("VALIDATION_MAX_CONCURRENT_ROLLOUT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| get_max_concurrent_rollout(num_gpus))
}

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
    config_nickname_generation: String,
    use_tool: bool,
    num_oneshot_epochs: usize,
    validation_rollout_secs: usize,
    training_hyperparameters: TrainingHyperparameters,
    oneshot_per_epoch_training_time: f32,
    num_iterations_limit: usize,
    num_gpus: usize,
    inference_backend: InferenceBackend,
    training_trajectory_len_cutoff: usize,
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
) {
    let client = Client::new();
    let oneshot_training_summary_parent_dir =
        training_summary_oneshot_parent_dir(mount_dir, model_cli_name, config_nickname_training);
    let oneshot_model_output_root =
        oneshot_epochs_parent_dir(mount_dir, model_cli_name, config_nickname_training);
    let training_epoch_stats =
        read_oneshot_training_epoch_stats(&oneshot_model_output_root, num_oneshot_epochs);
    let trained_epochs = detect_trained_oneshot_epochs(&oneshot_model_output_root, num_oneshot_epochs);

    let (already_validated_epochs, mut validation_accuracies) =
        read_existing_validation_summary(&oneshot_training_summary_parent_dir);

    let mut epochs_to_validate = Vec::new();
    if !already_validated_epochs.contains(&0) {
        epochs_to_validate.push(0);
    }
    epochs_to_validate.extend(
        trained_epochs
            .iter()
            .copied()
            .filter(|epoch| !already_validated_epochs.contains(epoch)),
    );

    if epochs_to_validate.is_empty() {
        log_info("No new oneshot epochs require validation; skipping validation phase");
        return;
    }

    log_info(format!(
        "Will validate epochs {:?} (trained epochs detected: {:?})",
        epochs_to_validate, trained_epochs
    ));
    best_effort_shutdown_stale_inference_wrapper().await;

    let launch_model_parent_dir = base_model_dir(mount_dir, model_cli_name);
    let launch_model_path = format!("{}/model", launch_model_parent_dir);
    let (sglang_port, mut handle) = launch_inference_wrapper::launch_inference_wrapper_process(
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
            max_concurrent_rollout: validation_max_concurrent_rollout(num_gpus),
        };
        let validation_summary =
            rollout_all::<M, Validation>(mount_dir, validation_program_config).await;
        log_info(format!(
            "Epoch {}: Validation rollout finished ({:.3}s, {} LLM calls)",
            epoch, validation_summary.elapsed_secs, validation_summary.total_llm_calls,
        ));

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

        write_training_summary(
            &oneshot_training_summary_parent_dir,
            epoch,
            num_oneshot_epochs,
            &validation_accuracies,
            &training_epoch_stats.throughputs,
            &training_epoch_stats.samples_trained,
            &training_epoch_stats.longest_non_oom_trajectory_lengths,
        );

        let validated_trained_epochs: Vec<usize> = trained_epochs
            .iter()
            .copied()
            .filter(|trained_epoch| validation_accuracies.contains_key(trained_epoch))
            .collect();
        prune_non_best_oneshot_models(
            &oneshot_model_output_root,
            &validated_trained_epochs,
            &validation_accuracies,
        );
    }

    log_info("Shutting down inference server after oneshot validation");
    let _ = handle.stop_signal_tx.send(true);
    shut_down_inference_wrapper_process(&mut handle.child).await;
    let _ = handle.listener_handle.await;
    log_info("Inference server shut down");
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
        use_tool,
        num_oneshot_epochs,
        validation_rollout_secs,
        num_gpus,
        inference_backend,
        mount_dir,
        generation_mount_dir,
        ..
    } = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));

    let process_title = format!(
        "oneshot_validation_{}_{}",
        model_cli_name, config_nickname_training
    );
    set_title(&process_title);
    check_sympy_availability().unwrap();
    configure_mount_dir(&mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));
    assert!(num_gpus > 0, "--num-gpus must be positive");
    assert!(
        num_oneshot_epochs > 0,
        "--num-oneshot-epochs must be positive"
    );

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
    ensure_parent_dir_exists(&text_log_summary_path)
        .unwrap_or_else(|err| panic!("failed to prepare validation summary log directory: {}", err));
    ensure_parent_dir_exists(&text_log_verbose_path)
        .unwrap_or_else(|err| panic!("failed to prepare validation verbose log directory: {}", err));
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
            run_oneshot_validation::<Gemma3_4BIt>(
                &model_cli_name,
                &config_nickname_training,
                &generation_mount_dir,
                &mount_dir,
                validation_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
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
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
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
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
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
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
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
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
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
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
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
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
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
                num_oneshot_epochs,
                validation_rollout_secs,
                num_gpus,
                inference_backend,
                &inference_wrapper_log_path,
                use_tool,
            )
            .await
        }
    }

    ProgressTextLogger::shutdown().await.unwrap();
}
