use std::{backtrace::Backtrace, path::Path};

use clap::{Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    chunked_judging::{
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT, DEFAULT_CACHE_VERSION,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL, judge_requests, read_judging_outputs,
    },
    constants,
    directories::{
        base_model_dir, inference_wrapper_log_path, oneshot_model_parent_dir, test_accuracy_path,
        text_logger_summary_path, text_logger_verbose_path, tree_artifacts_oneshot_path,
        tree_judgments_oneshot_path,
    },
    get_accuracy::TestAccuracyResult,
    hybrid_dataset::{DatasetSplit, Testing},
    json_toml_utils::{read_json, write_json},
    launch_inference_wrapper::{
        InferenceBackend, best_effort_shutdown_stale_inference_wrapper,
        launch_inference_wrapper_process, shut_down_inference_wrapper_process,
    },
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    rollout::{
        MultiTrialRolloutProgramConfig, RolloutProgramConfig, rollout_all, rollout_testing_trials,
    },
    rollout_config::RolloutConfig,
    tree_artifact::{TreeJudgment, read_marked_tree_artifact_chunks},
    tree_to_action::BranchingRuntimeOptions,
    utils::configure_mount_dir,
};
use ordered_float::NotNan;
use reqwest::Client;
use research_utility::progress_text_logger::ProgressTextLogger;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run test rollout and compute per-dataset accuracies with confidence intervals"
)]
struct Args {
    #[arg(long)]
    testing_configs_path: String,

    #[arg(long)]
    rollout_secs: usize,
    #[arg(long, default_value_t = 1)]
    num_gpus: usize,
    #[arg(long, default_value_t = InferenceBackend::Sglang)]
    inference_backend: InferenceBackend,
    #[arg(long, default_value = "all")]
    phase: TestingPhase,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TestingPhase {
    All,
    Rollout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestingConfig {
    model_cli_name: String,
    config_nickname: String,
    testing_rollout_config_path: String,
    #[serde(default)]
    num_rollout_trials: Option<usize>,
    epoch: usize,
    total_epochs: usize,
    mount_dir: String,
    use_tool: bool,
}

fn testing_trial_tree_artifact_path(base_path: &str, trial_index: usize) -> String {
    format!("{base_path}/trial_{trial_index}")
}

fn testing_trial_action_log_path(base_path: &str, trial_index: usize) -> String {
    format!(
        "{}/action_logs_testing.extsort",
        testing_trial_tree_artifact_path(base_path, trial_index)
    )
}

fn testing_trial_tree_judgment_path(base_path: &str, trial_index: usize) -> String {
    let path = Path::new(base_path);
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("tree_judgments_testing_oneshot");
    let extension = path.extension().and_then(|name| name.to_str());
    match extension {
        Some(extension) => parent
            .join(format!("{stem}_trial_{trial_index}.{extension}"))
            .to_string_lossy()
            .to_string(),
        None => parent
            .join(format!("{stem}_trial_{trial_index}"))
            .to_string_lossy()
            .to_string(),
    }
}

fn merge_test_accuracy_results(results: Vec<TestAccuracyResult>) -> TestAccuracyResult {
    let mut per_dataset = std::collections::BTreeMap::new();
    for result in results {
        for (dataset_name, dataset_accuracies) in result.per_dataset {
            per_dataset
                .entry(dataset_name)
                .or_insert_with(Vec::new)
                .extend(dataset_accuracies.accuracy_values);
        }
    }
    let per_dataset = per_dataset
        .into_iter()
        .map(|(dataset_name, accuracy_values)| {
            let n = accuracy_values.len() as f32;
            let mean = if n > 0.0 {
                accuracy_values.iter().sum::<f32>() / n
            } else {
                0.0
            };
            let variance = if n > 1.0 {
                accuracy_values
                    .iter()
                    .map(|a| (a - mean).powi(2))
                    .sum::<f32>()
                    / (n - 1.0)
            } else {
                0.0
            };
            let std_err = if n > 0.0 { (variance / n).sqrt() } else { 0.0 };
            (
                dataset_name,
                credit_assignment::get_accuracy::DatasetAccuracies {
                    accuracy_values,
                    mean_accuracy: mean,
                    confidence_interval_half_width: 1.96 * std_err,
                },
            )
        })
        .collect();
    let macro_average = credit_assignment::get_accuracy::equal_dataset_macro_average(&per_dataset);
    TestAccuracyResult {
        per_dataset,
        macro_average,
    }
}

fn model_cli_name_to_string(model_name: &LlmModelName) -> String {
    match model_name {
        LlmModelName::Gemma3_4b => Gemma3_4BIt::CLI_NAME,
        LlmModelName::Llama31_8b => Llama31_8BInstruct::CLI_NAME,
        LlmModelName::Mistral7bInstructV03 => Mistral7BInstructV03::CLI_NAME,
        LlmModelName::Qwen3_06b => Qwen3_06B::CLI_NAME,
        LlmModelName::Qwen3_4b => Qwen3_4B::CLI_NAME,
        LlmModelName::Qwen25_7b => Qwen25_7B::CLI_NAME,
        LlmModelName::Qwen35_08b => Qwen35_08B::CLI_NAME,
        LlmModelName::Qwen35_4b => Qwen35_4B::CLI_NAME,
    }
    .to_string()
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

async fn judge_testing_tree_artifacts<M: LlmModelMarker>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) {
    let artifacts = read_marked_tree_artifact_chunks::<M, Testing>(tree_artifact_path)
        .unwrap_or_else(|err| panic!("failed to read marked testing tree chunks: {}", err));
    let requests = artifacts
        .iter()
        .flat_map(|artifact| artifact.to_judging_requests(DEFAULT_CACHE_VERSION))
        .collect::<Vec<_>>();
    let judging_output_jsonl_path = format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/testing_judging_outputs.jsonl"
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
    .unwrap_or_else(|err| panic!("failed to judge testing tree artifacts: {}", err));
    println!("Testing judging summary: {summary:#?}");

    let outputs = read_judging_outputs(std::path::Path::new(&judging_output_jsonl_path))
        .unwrap_or_else(|err| panic!("failed to read testing judging outputs: {}", err));
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
                "failed to create testing tree judgment parent {}: {}",
                parent.display(),
                err
            )
        });
    }
    let mut file = std::fs::File::create(tree_judgment_jsonl_path).unwrap_or_else(|err| {
        panic!(
            "failed to create testing tree judgment JSONL {}: {}",
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
            Testing::dataset_file_postfix(),
            artifact.question.flat_id.0,
            outputs,
        )
        .unwrap_or_else(|err| panic!("failed to build testing tree judgment: {}", err));
        serde_json::to_writer(&mut file, &judgment)
            .unwrap_or_else(|err| panic!("failed to serialize testing tree judgment: {}", err));
        use std::io::Write;
        writeln!(file)
            .unwrap_or_else(|err| panic!("failed to write testing tree judgment: {}", err));
    }
}

async fn run_rollout_and_compute_accuracy<M: LlmModelMarker>(
    rollout_config: RolloutConfig<Testing>,
    testing_config: &TestingConfig,
    args: &Args,
    client: Option<Client>,
    posterior_calculation_config: PosteriorCalculationConfig,
    inference_endpoint: Option<InferenceEndpoint>,
) -> Option<TestAccuracyResult> {
    let tree_artifact_output_path = tree_artifacts_oneshot_path::<Testing>(
        &testing_config.mount_dir,
        M::CLI_NAME,
        &testing_config.config_nickname,
        testing_config.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to build testing tree artifact path: {}", err));
    let tree_judgment_jsonl_path = tree_judgments_oneshot_path::<Testing>(
        &testing_config.mount_dir,
        M::CLI_NAME,
        &testing_config.config_nickname,
        testing_config.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to build testing tree judgment path: {}", err));
    let num_rollout_trials = testing_config
        .num_rollout_trials
        .unwrap_or(rollout_config.num_trunks);
    assert!(
        num_rollout_trials > 0,
        "num_rollout_trials must be positive"
    );
    let use_explicit_trial_dirs = testing_config.num_rollout_trials.is_some();
    if matches!(args.phase, TestingPhase::All | TestingPhase::Rollout) {
        let client = client.expect("testing rollout phase requires reqwest client");
        let inference_endpoint =
            inference_endpoint.expect("testing rollout phase requires inference endpoint");
        if use_explicit_trial_dirs {
            let action_log_store_paths = (0..num_rollout_trials)
                .map(|trial_index| {
                    testing_trial_action_log_path(&tree_artifact_output_path, trial_index)
                })
                .collect::<Vec<_>>();
            let tree_artifact_output_paths = (0..num_rollout_trials)
                .map(|trial_index| {
                    testing_trial_tree_artifact_path(&tree_artifact_output_path, trial_index)
                })
                .collect::<Vec<_>>();
            let program_config = MultiTrialRolloutProgramConfig {
                config_nickname: testing_config.config_nickname.clone(),
                rollout_config: rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
                epoch: testing_config.epoch,
                client,
                inference_endpoint,
                rollout_secs: args.rollout_secs,
                finish_all_questions: false,
                total_epochs: testing_config.total_epochs,
                action_log_store_paths,
                tree_artifact_output_paths,
                use_tool: testing_config.use_tool,
                fixed_temperature: NotNan::new(constants::VALIDATION_TEMPERATURE).unwrap(),
                max_concurrent_rollout: 300,
                branching_options: BranchingRuntimeOptions::default(),
                tree_artifact_chunk_question_count: Some(DEFAULT_CACHE_CHUNK_QUESTION_COUNT),
            };
            let _ = rollout_testing_trials::<M, Testing>(&testing_config.mount_dir, program_config)
                .await;
        } else {
            let program_config = RolloutProgramConfig {
                config_nickname: testing_config.config_nickname.clone(),
                rollout_config: rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
                epoch: testing_config.epoch,
                client,
                inference_endpoint,
                rollout_secs: args.rollout_secs,
                finish_all_questions: false,
                total_epochs: testing_config.total_epochs,
                action_log_store_override_path: None,
                use_tool: testing_config.use_tool,
                fixed_temperature: NotNan::new(constants::VALIDATION_TEMPERATURE).unwrap(),
                max_concurrent_rollout: 300,
                branching_options: BranchingRuntimeOptions::default(),
                tree_artifact_output_path: Some(tree_artifact_output_path.clone()),
                tree_artifact_chunk_question_count: Some(DEFAULT_CACHE_CHUNK_QUESTION_COUNT),
                question_flat_id_start: None,
                question_flat_id_end: None,
                question_flat_ids: None,
            };
            let _ = rollout_all::<M, Testing>(&testing_config.mount_dir, program_config).await;
        }
    }

    None
}

async fn run_rollout_and_compute_accuracy_with_server<M: LlmModelMarker>(
    rollout_config: RolloutConfig<Testing>,
    testing_config: &TestingConfig,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
    inference_wrapper_log_path: &str,
) -> Result<Option<TestAccuracyResult>, String> {
    let mut launched_inference = None;
    if matches!(args.phase, TestingPhase::All | TestingPhase::Rollout) {
        best_effort_shutdown_stale_inference_wrapper().await;
        let model_parent_dir = if testing_config.epoch == 0 {
            base_model_dir(&testing_config.mount_dir, M::CLI_NAME)
        } else {
            oneshot_model_parent_dir(
                &testing_config.mount_dir,
                M::CLI_NAME,
                &testing_config.config_nickname,
                testing_config.epoch,
            )
        };
        let model_path = format!("{}/model", model_parent_dir);
        let (sglang_port, handle) = launch_inference_wrapper_process(
            args.inference_backend,
            &model_path,
            M::CLI_NAME,
            &testing_config.config_nickname,
            testing_config.epoch,
            M::API_NAME,
            args.num_gpus,
            inference_wrapper_log_path,
        )
        .await?;
        launched_inference = Some((sglang_port, handle));
    }

    let test_result = run_rollout_and_compute_accuracy::<M>(
        rollout_config,
        testing_config,
        args,
        if launched_inference.is_some() {
            Some(client)
        } else {
            None
        },
        posterior_calculation_config,
        launched_inference
            .as_ref()
            .map(|(port, _)| InferenceEndpoint::SglangPort(*port)),
    )
    .await;

    if let Some((_, mut handle)) = launched_inference {
        let _ = handle.stop_signal_tx.send(true);
        shut_down_inference_wrapper_process(&mut handle.child).await;
        let _ = handle.listener_handle.await;
    }
    Ok(test_result)
}

macro_rules! run_model_for_testing {
    ($model_name:expr, $rollout_config:expr, $testing_config:expr, $args:expr, $client:expr, $posterior:expr,
     $inference_wrapper_log_path:expr;
     $( $model_enum:path, $model_ty:ty ),+ $(,)?) => {{
        let model_name = $model_name;
        let rollout_config = $rollout_config;
        let testing_config = $testing_config;
        let args = $args;
        let client = $client;
        let posterior = $posterior;

        match model_name {
            $(
                $model_enum => {
                    run_rollout_and_compute_accuracy_with_server::<$model_ty>(
                        rollout_config,
                        testing_config,
                        args,
                        client,
                        posterior,
                        $inference_wrapper_log_path,
                    )
                    .await
                }
            ),+
        }
    }};
}

async fn run_testing_config(
    testing_config: &TestingConfig,
    args: &Args,
    client: Client,
) -> Result<(), String> {
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )?;
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&testing_config.model_cli_name, true)
        .map_err(|err| err.to_string())?;
    let model_cli_name = model_cli_name_to_string(&model_name);
    configure_mount_dir(&testing_config.mount_dir)?;
    let inference_wrapper_log_path = inference_wrapper_log_path(
        &testing_config.mount_dir,
        &model_cli_name,
        &testing_config.config_nickname,
    );
    let progress_text_summary_path = text_logger_summary_path(
        &testing_config.mount_dir,
        &model_cli_name,
        &testing_config.config_nickname,
    );
    let progress_text_verbose_path = text_logger_verbose_path(
        &testing_config.mount_dir,
        &model_cli_name,
        &testing_config.config_nickname,
    );
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .map_err(|err| format!("failed to prepare inference wrapper log directory: {}", err))?;
    ensure_parent_dir_exists(&progress_text_summary_path)
        .map_err(|err| format!("failed to prepare text log directory: {}", err))?;

    let rollout_config: RolloutConfig<Testing> =
        read_json::<RolloutConfig<Testing>>(&testing_config.testing_rollout_config_path)?;
    assert!(
        rollout_config.num_trunks == rollout_config.num_leaves,
        "num_trunks ({}) must equal num_leaves ({}) within each testing rollout trial (no branching)",
        rollout_config.num_trunks,
        rollout_config.num_leaves,
    );

    ProgressTextLogger::initialize(progress_text_summary_path, progress_text_verbose_path)
        .await
        .map_err(|err| err.to_string())?;

    let result = async {
        let maybe_test_result = run_model_for_testing!(
            model_name,
            rollout_config,
            testing_config,
            args,
            client,
            posterior_calculation_config,
            &inference_wrapper_log_path;
            LlmModelName::Qwen25_7b, Qwen25_7B,
            LlmModelName::Qwen3_06b, Qwen3_06B,
            LlmModelName::Qwen3_4b, Qwen3_4B,
            LlmModelName::Qwen35_4b, Qwen35_4B,
            LlmModelName::Qwen35_08b, Qwen35_08B,
            LlmModelName::Gemma3_4b, Gemma3_4BIt,
            LlmModelName::Llama31_8b, Llama31_8BInstruct,
            LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03
        )?;

        if let Some(test_result) = maybe_test_result {
            let output_path = test_accuracy_path(
                &testing_config.mount_dir,
                &model_cli_name,
                &testing_config.config_nickname,
                testing_config.epoch,
            );
            write_json(&output_path, &test_result)?;
            println!("Test accuracy results written to {}", output_path);
        } else {
            println!(
                "Testing phase {:?} completed without scoring output",
                args.phase
            );
        }
        Ok::<(), String>(())
    }
    .await;

    if let Err(err) = ProgressTextLogger::shutdown()
        .await
        .map_err(|err| err.to_string())
    {
        if result.is_ok() {
            return Err(err);
        }
        eprintln!(
            "warning: progress TUI shutdown failed after config failure: {}",
            err
        );
    }

    result
}

#[tokio::main]
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
    let args = Args::parse();
    check_sympy_availability().unwrap();
    assert!(args.num_gpus > 0, "num_gpus must be positive");

    println!("Starting test accuracy evaluation pipeline...");
    let client = Client::new();
    let testing_configs = read_json::<Vec<TestingConfig>>(&args.testing_configs_path)
        .unwrap_or_else(|err| panic!("failed to read testing configs: {}", err));
    assert!(
        !testing_configs.is_empty(),
        "testing_configs_path must contain at least one testing config"
    );
    if args.login_smoke {
        for testing_config in &testing_configs {
            LlmModelName::from_str(&testing_config.model_cli_name, true).unwrap_or_else(|err| {
                panic!(
                    "invalid model_cli_name in testing config {}: {}",
                    testing_config.config_nickname, err
                )
            });
            let _: RolloutConfig<Testing> = read_json(&testing_config.testing_rollout_config_path)
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to read testing rollout config for {}: {}",
                        testing_config.config_nickname, err
                    )
                });
            assert!(
                testing_config.total_epochs > 0,
                "testing config {} has non-positive total_epochs",
                testing_config.config_nickname
            );
        }
        println!(
            "login-smoke passed for bin_run_test: configs={}, phase={:?}, backend={:?}, num_gpus={}",
            testing_configs.len(),
            args.phase,
            args.inference_backend,
            args.num_gpus
        );
        return;
    }

    for (index, testing_config) in testing_configs.iter().enumerate() {
        println!(
            "Starting testing config {} of {}: model_cli_name={}, config_nickname={}, epoch={}, total_epochs={}, mount_dir={}",
            index + 1,
            testing_configs.len(),
            testing_config.model_cli_name,
            testing_config.config_nickname,
            testing_config.epoch,
            testing_config.total_epochs,
            testing_config.mount_dir,
        );
        if let Err(err) = run_testing_config(testing_config, &args, client.clone()).await {
            panic!(
                "testing config {} of {} failed for model_cli_name={}, config_nickname={}, epoch={}, total_epochs={}: {}",
                index + 1,
                testing_configs.len(),
                testing_config.model_cli_name,
                testing_config.config_nickname,
                testing_config.epoch,
                testing_config.total_epochs,
                err,
            );
        }
    }
}
