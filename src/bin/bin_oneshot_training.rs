use std::backtrace::Backtrace;
use std::collections::BTreeMap;
use std::path::Path;

use clap::{ArgAction, Parser, ValueEnum};
use proctitle::set_title;
use tokio::time::Instant;

use credit_assignment::{
    check_python_env::check_sympy_availability,
    direct_tool::{
        hybrid_dataset::{Training, Validation},
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
        rollout::{RolloutProgramConfig, rollout_all},
        rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
        training_set::generate_training_trajectories_with_path,
    },
    get_accuracy::get_accuracy_at_path,
    jinja_directories::{
        action_logs_oneshot_path_from_template, inference_wrapper_log_path_from_template,
        oneshot_model_checkpoint_dir_from_template, oneshot_model_parent_dir_from_template,
        training_summary_oneshot_parent_dir_from_template,
        training_trajectories_oneshot_path_from_template,
        training_trajectories_stats_oneshot_path_from_template,
        training_wrapper_log_path_from_template,
    },
    json_toml_utils::read_json,
    launch_inference_wrapper::{self, shut_down_inference_wrapper_process},
    launch_training_wrapper::run_training_wrapper_and_wait,
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    python_training_config::PythonTrainingConfig,
    utils::{configure_mount_dir, storage_large_files_dir},
};
use reqwest::Client;
use research_utility::progress_tui_logger::{
    ProgressTuiLogger, log_info, log_key_value_pair, log_state, log_warning,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "One-shot training: uses fixed rollout trajectories, trains all epochs first, then validates all models (including base model)"
)]
struct Args {
    #[arg(long)]
    model_cli_name: String,
    #[arg(long)]
    max_rollout_concurrency: usize,
    #[arg(long)]
    config_nickname_rollout: String,
    #[arg(long)]
    config_nickname_training: String,
    #[arg(long)]
    validation_rollout_config_path: String,
    #[arg(long)]
    training_rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    num_oneshot_epochs: usize,
    #[arg(long)]
    cumulative_avg_abs_advantage_cutoff: f32,
    #[arg(long, value_enum)]
    advantage_calculation_policy: AdvantageCalculationPolicy,
    #[arg(long)]
    training_config_common_path: String,
    #[arg(long)]
    oneshot_per_epoch_training_time: f32,
    #[arg(long)]
    num_iterations_limit: usize,
    #[arg(long)]
    validation_rollout_time_limit_secs: usize,
    #[arg(long, default_value_t = 1)]
    max_python_processes: usize,
    #[arg(long)]
    num_gpus: usize,
    #[arg(long)]
    mount_dir: String,
    #[arg(long, action = ArgAction::Set)]
    ui: bool,
    #[arg(long, action = ArgAction::Set)]
    positive_advantage_only: bool,
    #[arg(long, action = ArgAction::Set)]
    adam_fp32: bool,
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
    let accumulated_path = Path::new(summary_parent_dir).join("oneshot_aggregated_summary.json");
    let payload = serde_json::json!({
        "latest_epoch": latest_epoch,
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
    config_nickname_rollout: &str,
    config_nickname_training: &str,
    max_rollout_concurrency: usize,
    validation_rollout_config: DirectRolloutConfig<Validation>,
    training_set_rollout_config: DirectRolloutConfig<Training>,
    posterior_calculation_config: PosteriorCalculationConfig,
    num_oneshot_epochs: usize,
    validation_rollout_time_limit_secs: usize,
    max_python_processes: usize,
    cumulative_avg_abs_advantage_cutoff: f32,
    advantage_calculation_policy: AdvantageCalculationPolicy,
    training_config_common: credit_assignment::python_training_config::PythonTrainingConfigCommon,
    oneshot_per_epoch_training_time: f32,
    num_iterations_limit: usize,
    num_gpus: usize,
    inference_wrapper_log_path: &str,
    training_wrapper_log_path: &str,
    positive_advantage_only: bool,
    adam_fp32: bool,
) {
    let client = Client::new();

    // Resolve one-shot paths
    let oneshot_training_action_log_path = action_logs_oneshot_path_from_template::<Training>(
        model_cli_name,
        config_nickname_rollout,
        0,
    )
    .unwrap_or_else(|err| {
        panic!(
            "failed to resolve one-shot training action logs path: {}",
            err
        )
    });
    let oneshot_trajectories_dir =
        training_trajectories_oneshot_path_from_template(model_cli_name, config_nickname_training)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to resolve one-shot training trajectories path: {}",
                    err
                )
            });
    let oneshot_trajectories_msgpack_path = Path::new(&oneshot_trajectories_dir)
        .join("trajectories.msgpack")
        .to_string_lossy()
        .into_owned();
    let oneshot_stats_path = training_trajectories_stats_oneshot_path_from_template(
        model_cli_name,
        config_nickname_training,
    )
    .unwrap_or_else(|err| panic!("failed to resolve one-shot stats path: {}", err));
    let oneshot_config_bundle_path = Path::new(&oneshot_trajectories_dir)
        .join("config_bundle.json")
        .to_string_lossy()
        .into_owned();
    let oneshot_training_summary_parent_dir =
        training_summary_oneshot_parent_dir_from_template(model_cli_name, config_nickname_training)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to resolve one-shot training summary parent dir: {}",
                    err
                )
            });
    let artifact_root_dir = storage_large_files_dir()
        .unwrap_or_else(|err| panic!("failed to resolve artifact root dir: {}", err));

    // Step 1: Generate training trajectories from one-shot action logs
    log_state("Generating training trajectories from one-shot action logs");
    generate_training_trajectories_with_path::<M>(
        &oneshot_training_action_log_path,
        &oneshot_trajectories_dir,
        &oneshot_trajectories_msgpack_path,
        &oneshot_stats_path,
        &oneshot_config_bundle_path,
        training_set_rollout_config.clone(),
        posterior_calculation_config.clone(),
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
        positive_advantage_only,
    )
    .await;

    // ================================================================
    // Phase 1: Train all oneshot epochs in a single process
    //        (Adam state, training cursor, and adaptive batch state
    //         persist across epochs via checkpoint save/load)
    // ================================================================
    log_state(format!(
        "Starting oneshot training for {} epoch(s) in a single process",
        num_oneshot_epochs
    ));

    let shared_checkpoints_parent_dir =
        oneshot_model_checkpoint_dir_from_template(model_cli_name, config_nickname_training, 0)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to resolve shared oneshot checkpoints parent dir: {}",
                    err
                )
            });
    let oneshot_model_output_root = shared_checkpoints_parent_dir
        .rsplit_once('/')
        .map(|(parent, _)| parent.to_string())
        .unwrap_or_else(|| shared_checkpoints_parent_dir.clone());

    let base_model_parent_dir =
        oneshot_model_parent_dir_from_template(model_cli_name, config_nickname_training, 0)
            .unwrap_or_else(|err| {
                panic!("failed to resolve base oneshot model parent dir: {}", err)
            });

    let training_config = PythonTrainingConfig {
        common: training_config_common.clone(),
        training_time: oneshot_per_epoch_training_time,
        num_iterations_limit,
        artifact_root_dir: artifact_root_dir.clone(),
        hpc_training_root_dir: None,
        model_cli_name: model_cli_name.to_string(),
        config_nickname: config_nickname_training.to_string(),
        adam_fp32,
        epoch: 0,
        model_parent_dir: base_model_parent_dir,
        checkpoints_parent_dir: shared_checkpoints_parent_dir.clone(),
        final_model_output_parent_dir: shared_checkpoints_parent_dir.clone(),
        training_summary_parent_dir: shared_checkpoints_parent_dir.clone(),
        training_mode: "oneshot".to_string(),
        oneshot_num_epochs: num_oneshot_epochs,
        oneshot_model_output_root: oneshot_model_output_root.clone(),
    };

    let training_start_time = Instant::now();
    let mut training_throughputs: BTreeMap<usize, f32> = BTreeMap::new();
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
    for epoch in 0..num_oneshot_epochs {
        training_throughputs.insert(epoch, 0.0);
    }
    log_info(format!(
        "Oneshot training of {} epoch(s) finished in {:.3}s",
        num_oneshot_epochs, elapsed_secs
    ));

    log_state("All oneshot training epochs completed; starting validation phase");

    // ================================================================
    // Phase 2: Validate all models (base model + all trained epochs)
    // ================================================================
    let mut validation_accuracies: BTreeMap<usize, (f32, f32, f32, f32)> = BTreeMap::new();

    for epoch in 0..=num_oneshot_epochs {
        log_state(format!(
            "One-shot validation for epoch {}/{}",
            epoch, num_oneshot_epochs
        ));

        // --- Validate model ---
        let model_parent_dir =
            oneshot_model_parent_dir_from_template(model_cli_name, config_nickname_training, epoch)
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to resolve oneshot model parent dir for epoch {}: {}",
                        epoch, err
                    )
                });
        let model_path = format!("{}/model", model_parent_dir);

        log_info(format!(
            "Epoch {}: Launching inference server for model path {}",
            epoch, model_path
        ));
        let (sglang_port, mut inference_process, listener_stop_signal, listener_handle) =
            launch_inference_wrapper::launch_inference_wrapper_process(
                &model_path,
                model_cli_name,
                config_nickname_training,
                epoch,
                M::API_NAME,
                num_gpus,
                inference_wrapper_log_path,
            )
            .await
            .unwrap_or_else(|err| panic!("failed to launch inference server: {}", err));
        log_info(format!(
            "Epoch {}: Inference server listening on port {}",
            epoch, sglang_port
        ));

        // Run validation rollout to one-shot validation action logs
        let validation_action_log_path = action_logs_oneshot_path_from_template::<Validation>(
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
            max_rollout_concurrency,
            inference_endpoint: InferenceEndpoint::SglangPort(sglang_port),
            rollout_time_limit_secs: validation_rollout_time_limit_secs,
            max_python_processes,
            total_epochs: num_oneshot_epochs,
            action_log_store_override_path: Some(validation_action_log_path.clone()),
        };
        let validation_summary = rollout_all::<M, Validation>(validation_program_config).await;
        log_info(format!(
            "Epoch {}: Validation rollout finished ({:.3}s, {} LLM calls)",
            epoch, validation_summary.elapsed_secs, validation_summary.total_llm_calls,
        ));

        // Shut down inference server
        let _ = listener_stop_signal.send(true);
        shut_down_inference_wrapper_process(&mut inference_process).await;
        let _ = listener_handle.await;
        log_info(format!("Epoch {}: Inference server shut down", epoch));

        // Read validation accuracy from one-shot paths
        let accuracy_stats = get_accuracy_at_path::<M, Validation>(
            &validation_action_log_path,
            validation_rollout_config.clone(),
            posterior_calculation_config.clone(),
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
            &validation_accuracies,
            &training_throughputs,
        );
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
    let Args {
        model_cli_name,
        config_nickname_rollout,
        config_nickname_training,
        max_rollout_concurrency,
        validation_rollout_config_path,
        training_rollout_config_path,
        posterior_hyperparameters_path,
        num_oneshot_epochs,
        ui,
        validation_rollout_time_limit_secs,
        max_python_processes,
        cumulative_avg_abs_advantage_cutoff,
        advantage_calculation_policy,
        training_config_common_path,
        oneshot_per_epoch_training_time,
        num_iterations_limit,
        num_gpus,
        mount_dir,
        positive_advantage_only,
        adam_fp32,
    } = Args::parse();
    let process_title = format!(
        "oneshot_training_{}_{}",
        model_cli_name, config_nickname_training
    );
    set_title(&process_title);
    check_sympy_availability().unwrap();
    assert!(
        max_python_processes > 0,
        "max_python_processes must be positive"
    );
    configure_mount_dir(&mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));
    let inference_wrapper_log_path =
        inference_wrapper_log_path_from_template(&model_cli_name, &config_nickname_training)
            .unwrap_or_else(|err| panic!("failed to render inference wrapper log path: {}", err));
    let training_wrapper_log_path =
        training_wrapper_log_path_from_template(&model_cli_name, &config_nickname_training)
            .unwrap_or_else(|err| panic!("failed to render training wrapper log path: {}", err));
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

    if ui {
        let tui_log_path = format!(
            "progress_tui_log_oneshot_training_{}_{}.bin",
            model_cli_name, config_nickname_training
        );
        ProgressTuiLogger::initialize(tui_log_path).await.unwrap();
    }
    let validation_rollout_config: DirectRolloutConfig<Validation> =
        read_json(&validation_rollout_config_path).unwrap();
    let training_set_rollout_config: DirectRolloutConfig<Training> =
        read_json(&training_rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(&posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let training_config_common: credit_assignment::python_training_config::PythonTrainingConfigCommon =
        credit_assignment::json_toml_utils::read_toml(&training_config_common_path).unwrap();
    let model_name = LlmModelName::from_str(&model_cli_name, true).unwrap();

    match model_name {
        LlmModelName::Gemma3_4b => {
            run_oneshot_training::<Gemma3_4BIt>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
        LlmModelName::Llama31_8b => {
            run_oneshot_training::<Llama31_8BInstruct>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
        LlmModelName::Mistral7bInstructV03 => {
            run_oneshot_training::<Mistral7BInstructV03>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
        LlmModelName::Qwen3_06b => {
            run_oneshot_training::<Qwen3_06B>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
        LlmModelName::Qwen3_4b => {
            run_oneshot_training::<Qwen3_4B>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
        LlmModelName::Qwen25_7b => {
            run_oneshot_training::<Qwen25_7B>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
        LlmModelName::Qwen35_08b => {
            run_oneshot_training::<Qwen35_08B>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
        LlmModelName::Qwen35_4b => {
            run_oneshot_training::<Qwen35_4B>(
                &model_cli_name,
                &config_nickname_rollout,
                &config_nickname_training,
                max_rollout_concurrency,
                validation_rollout_config,
                training_set_rollout_config,
                posterior_calculation_config,
                num_oneshot_epochs,
                validation_rollout_time_limit_secs,
                max_python_processes,
                cumulative_avg_abs_advantage_cutoff,
                advantage_calculation_policy,
                training_config_common,
                oneshot_per_epoch_training_time,
                num_iterations_limit,
                num_gpus,
                &inference_wrapper_log_path,
                &training_wrapper_log_path,
                positive_advantage_only,
                adam_fp32,
            )
            .await
        }
    }

    if ui {
        ProgressTuiLogger::shutdown().await.unwrap();
    }
}
