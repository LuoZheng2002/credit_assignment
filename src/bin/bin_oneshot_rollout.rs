use std::backtrace::Backtrace;
use std::path::Path;

use clap::{Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    constants,
    constants::get_max_concurrent_rollout,
    directories::{
        action_logs_oneshot_path, base_model_dir, inference_wrapper_log_path, model_parent_dir,
        rollout_summary_oneshot_path, text_logger_summary_path, text_logger_verbose_path,
        tree_artifacts_oneshot_chunk_done_path, tree_artifacts_oneshot_path,
    },
    hybrid_dataset::{DatasetSplit, DatasetSplitEnum, Testing, Training, Validation},
    json_toml_utils::read_json,
    launch_inference_wrapper::{
        InferenceBackend, best_effort_shutdown_stale_inference_wrapper,
        launch_inference_wrapper_process, shut_down_inference_wrapper_process,
    },
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelMarker, LlmModelName, Mistral7BInstructV03,
        Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::{BranchingPolicy, RolloutConfig},
    tree_action_log::ActionLogStore,
    tree_to_action::BranchingRuntimeOptions,
    utils::configure_mount_dir,
};
use reqwest::Client;
use research_utility::progress_text_logger::{ProgressTextLogger, log_info};
use serde::Deserialize;

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(short = 'c', long)]
    config_path: String,
    #[arg(long, default_value_t = false)]
    enable_uncertainty_aware_branching: bool,
    #[arg(long, default_value_t = false)]
    force_selected_branch_token: bool,
    #[arg(long)]
    num_questions_per_chunk: Option<usize>,
    #[arg(long)]
    num_chunks: Option<usize>,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Args {
    model_cli_name: String,
    config_nickname_rollout: String,
    rollout_config_path: String,
    use_tool: bool,
    dataset_split: DatasetSplitEnum,
    #[serde(default)]
    epoch: usize,
    #[serde(default = "default_total_epochs")]
    total_epochs: usize,
    rollout_secs: usize,
    mount_dir: String,
    num_gpus: usize,
    inference_backend: InferenceBackend,
    #[serde(default)]
    total_time_limit_hours: f32,
    #[serde(default)]
    enable_uncertainty_aware_branching: bool,
    #[serde(default)]
    force_selected_branch_token: bool,
    #[serde(default)]
    num_questions_per_chunk: Option<usize>,
    #[serde(default)]
    num_chunks: Option<usize>,
}

fn default_total_epochs() -> usize {
    1
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

fn remove_path_if_exists(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
    .map_err(|err| format!("failed to remove stale path {}: {}", path.display(), err))
}

fn remove_stale_action_log_store(action_log_store_path: &str) -> Result<(), String> {
    let base_path = Path::new(action_log_store_path);
    remove_path_if_exists(base_path)?;
    remove_path_if_exists(&base_path.with_extension("config_bundle.json"))?;
    remove_path_if_exists(&base_path.with_extension("elapsed_time.txt"))?;
    Ok(())
}

async fn run_rollout_for_split<M: LlmModelMarker, S: DatasetSplit>(
    rollout_config: RolloutConfig<S>,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
    num_gpus: usize,
    inference_backend: InferenceBackend,
    inference_wrapper_log_path: &str,
    branching_options: BranchingRuntimeOptions,
) {
    let branching_options = BranchingRuntimeOptions {
        tree_rl_entropy_guided_branching: rollout_config.branching_policy
            == BranchingPolicy::TreeRlEntropyGuided,
        ..branching_options
    };
    let action_log_store_override_path = action_logs_oneshot_path::<S>(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
        args.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to resolve one-shot action logs path: {}", err));
    let tree_artifact_output_path = tree_artifacts_oneshot_path::<S>(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
        args.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to resolve one-shot tree artifacts path: {}", err));
    let (question_flat_id_start, question_flat_id_end, chunk_indices_to_check) = if S::IS_TRAINING {
        match args.num_questions_per_chunk {
            Some(num_questions_per_chunk) => {
                assert!(
                    num_questions_per_chunk > 0,
                    "num_questions_per_chunk must be positive"
                );
                let num_chunks = args.num_chunks.unwrap_or(1);
                assert!(num_chunks > 0, "num_chunks must be positive");
                let end = num_questions_per_chunk
                    .checked_mul(num_chunks)
                    .expect("num_questions_per_chunk * num_chunks overflowed");
                log_info(format!(
                    "Using deterministic training rollout chunk range: flat_id=[0, {}) from num_questions_per_chunk={} num_chunks={}",
                    end, num_questions_per_chunk, num_chunks
                ));
                (Some(0), Some(end), (0..num_chunks).collect::<Vec<_>>())
            }
            None => {
                assert!(
                    args.num_chunks.is_none(),
                    "num_chunks requires num_questions_per_chunk"
                );
                (None, None, vec![0])
            }
        }
    } else {
        assert!(
            args.num_questions_per_chunk.is_none() && args.num_chunks.is_none(),
            "question chunk slicing is only supported for training rollout"
        );
        (None, None, vec![0])
    };
    let all_target_chunks_done = chunk_indices_to_check.iter().all(|chunk_index| {
        Path::new(&tree_artifacts_oneshot_chunk_done_path(
            &tree_artifact_output_path,
            *chunk_index,
        ))
        .exists()
    });
    if !all_target_chunks_done {
        remove_stale_action_log_store(&action_log_store_override_path)
            .unwrap_or_else(|err| panic!("failed to remove stale action logs: {}", err));
    }

    // Check if previous run already exhausted the time limit; if so, skip the
    // inference server startup and rollout entirely.
    let prev_elapsed =
        ActionLogStore::<M, S>::initialize_if_missing(&action_log_store_override_path)
            .and_then(|store| store.read_elapsed_time())
            .unwrap_or(0.0);
    if prev_elapsed >= args.rollout_secs as f32 {
        log_info(format!(
            "Skipping rollout: previous elapsed time ({prev_elapsed:.1}s) already meets \
             or exceeds the time limit ({}s). No new actions will be added.",
            args.rollout_secs
        ));
        return;
    }

    best_effort_shutdown_stale_inference_wrapper().await;
    let model_parent_dir = if args.epoch == 0 {
        base_model_dir(&args.mount_dir, M::CLI_NAME)
    } else {
        model_parent_dir(
            &args.mount_dir,
            M::CLI_NAME,
            &args.config_nickname_rollout,
            args.epoch,
        )
    };
    let model_path = format!("{}/model", model_parent_dir);

    let (sglang_port, mut handle) = launch_inference_wrapper_process(
        inference_backend,
        &model_path,
        M::CLI_NAME,
        &args.config_nickname_rollout,
        args.epoch,
        M::API_NAME,
        num_gpus,
        inference_wrapper_log_path,
    )
    .await
    .unwrap_or_else(|err| panic!("failed to launch inference server: {}", err));

    let program_config = RolloutProgramConfig {
        config_nickname: args.config_nickname_rollout.clone(),
        rollout_config,
        posterior_calculation_config,
        epoch: args.epoch,
        client,
        inference_endpoint: credit_assignment::llm_model::InferenceEndpoint::SglangPort(
            sglang_port,
        ),
        rollout_secs: args.rollout_secs,
        finish_all_questions: false,
        total_epochs: args.total_epochs,
        action_log_store_override_path: Some(action_log_store_override_path),
        use_tool: args.use_tool,
        fixed_temperature: constants::temperature_by_split::<S>(),
        max_concurrent_rollout: get_max_concurrent_rollout(num_gpus),
        branching_options,
        tree_artifact_output_path: Some(tree_artifact_output_path),
        tree_artifact_chunk_question_count: args.num_questions_per_chunk,
        question_flat_id_start,
        question_flat_id_end,
        question_flat_ids: None,
    };
    let summary = rollout_all::<M, S>(&args.mount_dir, program_config).await;

    let _ = handle.stop_signal_tx.send(true);
    shut_down_inference_wrapper_process(&mut handle.child).await;
    let _ = handle.listener_handle.await;

    let summary_path = rollout_summary_oneshot_path(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
    );
    if let Some(parent) = std::path::Path::new(&summary_path).parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("failed to create dir {}: {}", parent.display(), err));
    }
    let json = serde_json::to_string_pretty(&summary)
        .unwrap_or_else(|err| panic!("failed to serialize rollout summary: {}", err));
    std::fs::write(&summary_path, json).unwrap_or_else(|err| {
        panic!(
            "failed to write rollout summary to {}: {}",
            summary_path, err
        )
    });
    println!("Rollout summary written to {}", summary_path);
}

macro_rules! run_rollout {
    (
        $model_name:expr,
        $dataset_split:expr,
        $args:expr,
        $client:expr,
        $posterior:expr,
        $num_gpus:expr,
        $inference_backend:expr,
        $log_path:expr;
        $branching_options:expr;
        $( $model_enum:path, $model_ty:ty ),+ $(,)?;
        $( $split_enum:path, $split_ty:ty ),+ $(,)?
    ) => {{
        let model_name = $model_name;
        let dataset_split = $dataset_split;
        let args = $args;
        let client = $client;
        let posterior = $posterior;
        let num_gpus = $num_gpus;
        let inference_backend = $inference_backend;
        let log_path = $log_path;
        let branching_options = $branching_options;

        macro_rules! run_model_for_split {
            ($rollout_config:expr, $inner_split_ty:ty) => {
                match model_name {
                    $(
                        $model_enum => {
                            run_rollout_for_split::<$model_ty, $inner_split_ty>(
                                $rollout_config,
                                args,
                                client,
                                posterior,
                                num_gpus,
                                inference_backend,
                                log_path,
                                branching_options,
                            )
                            .await
                        }
                    ),+
                }
            };
        }

        match dataset_split {
            $(
                $split_enum => {
                    let rollout_config: RolloutConfig<$split_ty> =
                        read_json(&args.rollout_config_path).unwrap();
                    run_model_for_split!(rollout_config, $split_ty)
                }
            ),+
        }
    }};
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
    let CliArgs {
        config_path,
        enable_uncertainty_aware_branching,
        force_selected_branch_token,
        num_questions_per_chunk,
        num_chunks,
        login_smoke,
    } = CliArgs::parse();
    let config_contents = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read config file '{}': {}", config_path, err));
    let mut args: Args = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));
    if num_questions_per_chunk.is_some() {
        args.num_questions_per_chunk = num_questions_per_chunk;
    }
    if num_chunks.is_some() {
        args.num_chunks = num_chunks;
    }
    check_sympy_availability().unwrap();
    assert!(args.total_epochs > 0, "total_epochs must be positive");
    assert!(args.num_gpus > 0, "--num-gpus must be positive");
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )
    .unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&args.model_cli_name, true).unwrap();
    let branching_options = BranchingRuntimeOptions {
        uncertainty_aware_branching: args.enable_uncertainty_aware_branching
            || enable_uncertainty_aware_branching,
        force_selected_branch_token: args.force_selected_branch_token
            || force_selected_branch_token,
        tree_rl_entropy_guided_branching: false,
    };
    match args.dataset_split {
        DatasetSplitEnum::Training => {
            let _: RolloutConfig<Training> = read_json(&args.rollout_config_path).unwrap();
        }
        DatasetSplitEnum::Validation => {
            let _: RolloutConfig<Validation> = read_json(&args.rollout_config_path).unwrap();
        }
        DatasetSplitEnum::Testing => {
            let _: RolloutConfig<Testing> = read_json(&args.rollout_config_path).unwrap();
        }
    }
    if login_smoke {
        println!(
            "login-smoke passed for bin_oneshot_rollout: model={}, config={}, split={:?}, backend={:?}, num_gpus={}",
            args.model_cli_name,
            args.config_nickname_rollout,
            args.dataset_split,
            args.inference_backend,
            args.num_gpus
        );
        return;
    }
    configure_mount_dir(&args.mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));

    println!("Starting one-shot rollout pipeline...");
    let client = Client::new();
    log_info(format!(
        "branching_options uncertainty_aware_branching={} force_selected_branch_token={} tree_rl_entropy_guided_branching={}",
        branching_options.uncertainty_aware_branching,
        branching_options.force_selected_branch_token,
        branching_options.tree_rl_entropy_guided_branching
    ));

    let inference_wrapper_log_path = inference_wrapper_log_path(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
    );
    ensure_parent_dir_exists(&inference_wrapper_log_path)
        .unwrap_or_else(|err| panic!("failed to prepare inference wrapper log directory: {}", err));

    let text_log_summary_path = text_logger_summary_path(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
    );
    let text_log_verbose_path = text_logger_verbose_path(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
    );
    ProgressTextLogger::initialize(text_log_summary_path, text_log_verbose_path)
        .await
        .unwrap();

    run_rollout!(
        model_name,
        args.dataset_split,
        &args,
        client,
        posterior_calculation_config,
        args.num_gpus,
        args.inference_backend,
        &inference_wrapper_log_path;
        branching_options;
        LlmModelName::Qwen25_7b, Qwen25_7B,
        LlmModelName::Qwen3_06b, Qwen3_06B,
        LlmModelName::Qwen3_4b, Qwen3_4B,
        LlmModelName::Qwen35_4b, Qwen35_4B,
        LlmModelName::Qwen35_08b, Qwen35_08B,
        LlmModelName::Gemma3_4b, Gemma3_4BIt,
        LlmModelName::Llama31_8b, Llama31_8BInstruct,
        LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03;
        DatasetSplitEnum::Training, Training,
        DatasetSplitEnum::Validation, Validation,
        DatasetSplitEnum::Testing, Testing
    );

    ProgressTextLogger::shutdown().await.unwrap();
}
