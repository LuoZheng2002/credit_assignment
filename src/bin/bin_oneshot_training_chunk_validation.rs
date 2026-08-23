use std::{backtrace::Backtrace, collections::BTreeSet, path::Path};

use clap::{Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    chunked_judging::{
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT, DEFAULT_CACHE_VERSION,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL, judge_requests, read_judging_outputs,
    },
    directories::{
        base_model_dir, inference_wrapper_log_path, oneshot_model_parent_dir,
        training_trajectories_oneshot_chunk_path, training_trajectories_oneshot_path,
        tree_artifacts_oneshot_path, tree_judgments_oneshot_path,
    },
    get_accuracy::get_accuracy_from_tree_judgments_at_path,
    hybrid_dataset::{DatasetSplit, Training},
    json_toml_utils::{read_json, write_json},
    launch_inference_wrapper::{
        self, InferenceBackend, best_effort_shutdown_stale_inference_wrapper,
        shut_down_inference_wrapper_process, update_inference_model,
    },
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    oneshot_training_summary::{derive_phase_log_path, ensure_parent_dir_exists},
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    python_training_config::TrainingHyperparameters,
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::RolloutConfig,
    training_set::open_training_trajectories_file,
    tree_artifact::{TreeJudgment, read_marked_tree_artifact_chunks},
    tree_to_action::BranchingRuntimeOptions,
    utils::configure_mount_dir,
};
use ordered_float::NotNan;
use proctitle::set_title;
use reqwest::Client;
use research_utility::progress_text_logger::{ProgressTextLogger, log_info};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(short = 'c', long)]
    config_path: String,
    #[arg(long)]
    chunk_index: Option<usize>,
    #[arg(long)]
    all_chunks: bool,
    #[arg(long, default_value = "all")]
    phase: ChunkValidationPhase,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ChunkValidationPhase {
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
    #[serde(default)]
    validation_num_rollout_trials: Option<usize>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingChunkValidationStats {
    chunk_index: usize,
    observed_num_trajectories: usize,
    observed_num_questions: usize,
    min_observed_flat_id: Option<usize>,
    max_observed_flat_id: Option<usize>,
    before_epoch: usize,
    after_epoch: usize,
    before_accuracy: Option<(f32, f32, f32, f32)>,
    after_accuracy: Option<(f32, f32, f32, f32)>,
    before_tree_artifact_path: String,
    after_tree_artifact_path: String,
    before_tree_judgment_path: String,
    after_tree_judgment_path: String,
}

fn chunk_validation_config_nickname(config_nickname_training: &str, chunk_index: usize) -> String {
    format!("{config_nickname_training}_training_chunk_{chunk_index}")
}

fn observed_training_chunk_question_ids<M: LlmModelMarker>(
    chunk_path: &str,
) -> (usize, BTreeSet<usize>) {
    let trajectories = open_training_trajectories_file::<M>(chunk_path);
    let ids = trajectories
        .iter()
        .map(|trajectory| trajectory.question.flat_id.0)
        .collect::<BTreeSet<_>>();
    (trajectories.len(), ids)
}

fn write_observed_stats(
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname_training: &str,
    chunk_index: usize,
    stats: &TrainingChunkValidationStats,
) {
    let output_path = format!(
        "{mount_dir}/small_files/{model_cli_name}/{config_nickname_training}/training_chunk_validation/chunk_{chunk_index}.json"
    );
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "failed to create training chunk validation stats parent {}: {}",
                parent.display(),
                err
            )
        });
    }
    write_json(output_path, stats).unwrap();
}

async fn judge_tree_artifacts<M: LlmModelMarker>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    mount_dir: &str,
    model_cli_name: &str,
    config_nickname: &str,
    epoch: usize,
) {
    let artifacts = read_marked_tree_artifact_chunks::<M, Training>(tree_artifact_path)
        .unwrap_or_else(|err| panic!("failed to read marked chunk validation trees: {}", err));
    let requests = artifacts
        .iter()
        .flat_map(|artifact| artifact.to_judging_requests(DEFAULT_CACHE_VERSION))
        .collect::<Vec<_>>();
    let judging_output_jsonl_path = format!(
        "{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/epoch_{epoch}/training_chunk_validation_judging_outputs.jsonl"
    );
    let cache_dir =
        format!("{mount_dir}/medium_files/{model_cli_name}/{config_nickname}/judgment_cache");
    let escalation_jsonl_path = format!(
        "{mount_dir}/small_files/{model_cli_name}/{config_nickname}/judgment_escalations.jsonl"
    );
    let summary = judge_requests(
        requests,
        Path::new(&judging_output_jsonl_path),
        Path::new(&cache_dir),
        Path::new(&escalation_jsonl_path),
        DEFAULT_CACHE_VERSION,
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL,
    )
    .await
    .unwrap_or_else(|err| panic!("failed to judge training chunk validation trees: {}", err));
    log_info(format!(
        "Training chunk validation judging summary: {summary:#?}"
    ));

    let outputs = read_judging_outputs(Path::new(&judging_output_jsonl_path))
        .unwrap_or_else(|err| panic!("failed to read training chunk judging outputs: {}", err));
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
    if let Some(parent) = Path::new(tree_judgment_jsonl_path).parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "failed to create training chunk judgment parent {}: {}",
                parent.display(),
                err
            )
        });
    }
    let mut file = std::fs::File::create(tree_judgment_jsonl_path).unwrap_or_else(|err| {
        panic!(
            "failed to create training chunk judgment JSONL {}: {}",
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
            Training::dataset_file_postfix(),
            artifact.question.flat_id.0,
            outputs,
        )
        .unwrap_or_else(|err| panic!("failed to build training chunk judgment: {}", err));
        serde_json::to_writer(&mut file, &judgment)
            .unwrap_or_else(|err| panic!("failed to serialize training chunk judgment: {}", err));
        use std::io::Write;
        writeln!(file).unwrap_or_else(|err| panic!("failed to write chunk judgment: {}", err));
    }
}

async fn run_chunk_validation<M: LlmModelMarker>(
    args: &Args,
    chunk_index: usize,
    phase: ChunkValidationPhase,
) {
    configure_mount_dir(&args.mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));
    configure_mount_dir(&args.generation_mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure generation mount dir: {}", err));
    let trajectories_dir = training_trajectories_oneshot_path(
        &args.generation_mount_dir,
        &args.model_cli_name,
        &args.config_nickname_generation,
    );
    let chunk_path = training_trajectories_oneshot_chunk_path(&trajectories_dir, chunk_index);
    assert!(
        Path::new(&chunk_path).is_file(),
        "training trajectory chunk not found: {}",
        chunk_path
    );
    let (observed_num_trajectories, observed_ids) =
        observed_training_chunk_question_ids::<M>(&chunk_path);
    assert!(
        !observed_ids.is_empty(),
        "training chunk contains no observed questions: {}",
        chunk_path
    );
    let min_flat_id = *observed_ids.iter().next().unwrap();
    let max_flat_id = *observed_ids.iter().next_back().unwrap();
    let before_epoch = chunk_index;
    let after_epoch = chunk_index + 1;
    assert!(
        after_epoch <= args.num_oneshot_epochs,
        "chunk_index {} maps to after_epoch {}, beyond num_oneshot_epochs {}",
        chunk_index,
        after_epoch,
        args.num_oneshot_epochs
    );
    let diagnostic_config_nickname =
        chunk_validation_config_nickname(&args.config_nickname_training, chunk_index);
    let before_config_nickname =
        format!("{diagnostic_config_nickname}_before_epoch_{before_epoch}");
    let after_config_nickname = format!("{diagnostic_config_nickname}_after_epoch_{after_epoch}");
    let before_tree_artifact_path = tree_artifacts_oneshot_path::<Training>(
        &args.mount_dir,
        M::CLI_NAME,
        &before_config_nickname,
        before_epoch,
    )
    .unwrap();
    let after_tree_artifact_path = tree_artifacts_oneshot_path::<Training>(
        &args.mount_dir,
        M::CLI_NAME,
        &after_config_nickname,
        after_epoch,
    )
    .unwrap();
    let before_tree_judgment_path = tree_judgments_oneshot_path::<Training>(
        &args.mount_dir,
        M::CLI_NAME,
        &before_config_nickname,
        before_epoch,
    )
    .unwrap();
    let after_tree_judgment_path = tree_judgments_oneshot_path::<Training>(
        &args.mount_dir,
        M::CLI_NAME,
        &after_config_nickname,
        after_epoch,
    )
    .unwrap();

    let validation_rollout_config: RolloutConfig<Training> =
        read_json(credit_assignment::directories::VALIDATION_ROLLOUT_CONFIG_PATH).unwrap();
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )
    .unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    if matches!(
        phase,
        ChunkValidationPhase::All | ChunkValidationPhase::Rollout
    ) {
        best_effort_shutdown_stale_inference_wrapper().await;
        let client = Client::new();
        let launch_model_path = format!("{}/model", base_model_dir(&args.mount_dir, M::CLI_NAME));
        let inference_log_path = derive_phase_log_path(
            &inference_wrapper_log_path(&args.mount_dir, M::CLI_NAME, &diagnostic_config_nickname),
            "training_chunk_validation",
        );
        ensure_parent_dir_exists(&inference_log_path).unwrap();
        let (port, mut handle) = launch_inference_wrapper::launch_inference_wrapper_process(
            args.inference_backend,
            &launch_model_path,
            M::CLI_NAME,
            &diagnostic_config_nickname,
            0,
            M::API_NAME,
            args.num_gpus,
            &inference_log_path,
        )
        .await
        .unwrap_or_else(|err| panic!("failed to launch inference server: {}", err));
        for (epoch, output_path, rollout_config_nickname) in [
            (
                before_epoch,
                before_tree_artifact_path.clone(),
                before_config_nickname.clone(),
            ),
            (
                after_epoch,
                after_tree_artifact_path.clone(),
                after_config_nickname.clone(),
            ),
        ] {
            log_info(format!(
                "Training chunk validation preparing rollout for chunk {} epoch {} config={} output_path={}",
                chunk_index, epoch, rollout_config_nickname, output_path
            ));
            if epoch > 0 {
                let model_path = format!(
                    "{}/model",
                    oneshot_model_parent_dir(
                        &args.mount_dir,
                        M::CLI_NAME,
                        &args.config_nickname_training,
                        epoch,
                    )
                );
                update_inference_model(port, &model_path, &inference_log_path)
                    .await
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to update inference model to epoch {}: {}",
                            epoch, err
                        )
                    });
            }
            log_info(format!(
                "Training chunk validation entering rollout_all for chunk {} epoch {} config={}",
                chunk_index, epoch, rollout_config_nickname
            ));
            let program_config = RolloutProgramConfig {
                config_nickname: rollout_config_nickname,
                rollout_config: validation_rollout_config.clone(),
                posterior_calculation_config: posterior_calculation_config.clone(),
                epoch,
                client: client.clone(),
                inference_endpoint: InferenceEndpoint::SglangPort(port),
                rollout_secs: args.validation_rollout_secs,
                finish_all_questions: true,
                total_epochs: args.num_oneshot_epochs + 1,
                action_log_store_override_path: None,
                use_tool: args.use_tool,
                fixed_temperature: NotNan::new(0.0).unwrap(),
                max_concurrent_rollout: 200,
                branching_options: BranchingRuntimeOptions::default(),
                tree_artifact_output_path: Some(output_path),
                tree_artifact_chunk_question_count: None,
                question_flat_id_start: Some(min_flat_id),
                question_flat_id_end: Some(max_flat_id + 1),
                question_flat_ids: Some(observed_ids.clone()),
            };
            let summary = rollout_all::<M, Training>(&args.mount_dir, program_config).await;
            log_info(format!(
                "Training chunk validation rollout epoch {} finished ({:.3}s, {} LLM calls)",
                epoch, summary.elapsed_secs, summary.total_llm_calls
            ));
        }
        let _ = handle.stop_signal_tx.send(true);
        shut_down_inference_wrapper_process(&mut handle.child).await;
        let _ = handle.listener_handle.await;
    }

    if matches!(
        phase,
        ChunkValidationPhase::All | ChunkValidationPhase::Judge
    ) {
        judge_tree_artifacts::<M>(
            &before_tree_artifact_path,
            &before_tree_judgment_path,
            &args.mount_dir,
            M::CLI_NAME,
            &before_config_nickname,
            before_epoch,
        )
        .await;
        judge_tree_artifacts::<M>(
            &after_tree_artifact_path,
            &after_tree_judgment_path,
            &args.mount_dir,
            M::CLI_NAME,
            &after_config_nickname,
            after_epoch,
        )
        .await;
    }

    let mut before_accuracy = None;
    let mut after_accuracy = None;
    if matches!(
        phase,
        ChunkValidationPhase::All | ChunkValidationPhase::Score
    ) {
        before_accuracy = get_accuracy_from_tree_judgments_at_path::<M, Training>(
            &before_tree_artifact_path,
            &before_tree_judgment_path,
            "Training chunk before accuracy",
        )
        .await
        .accuracy_tuple();
        after_accuracy = get_accuracy_from_tree_judgments_at_path::<M, Training>(
            &after_tree_artifact_path,
            &after_tree_judgment_path,
            "Training chunk after accuracy",
        )
        .await
        .accuracy_tuple();
    }

    let stats = TrainingChunkValidationStats {
        chunk_index,
        observed_num_trajectories,
        observed_num_questions: observed_ids.len(),
        min_observed_flat_id: Some(min_flat_id),
        max_observed_flat_id: Some(max_flat_id),
        before_epoch,
        after_epoch,
        before_accuracy,
        after_accuracy,
        before_tree_artifact_path,
        after_tree_artifact_path,
        before_tree_judgment_path,
        after_tree_judgment_path,
    };
    write_observed_stats(
        &args.mount_dir,
        M::CLI_NAME,
        &args.config_nickname_training,
        chunk_index,
        &stats,
    );
}

async fn run_selected_chunk_validations<M: LlmModelMarker>(
    args: &Args,
    chunk_index: Option<usize>,
    all_chunks: bool,
    phase: ChunkValidationPhase,
) {
    assert!(
        chunk_index.is_none() || !all_chunks,
        "pass either --chunk-index or --all-chunks, not both"
    );
    let chunk_indices = if let Some(chunk_index) = chunk_index {
        vec![chunk_index]
    } else {
        (0..args.num_oneshot_epochs).collect::<Vec<_>>()
    };
    assert!(
        all_chunks || chunk_index.is_some(),
        "no chunk selector was provided; use --all-chunks or --chunk-index"
    );
    for chunk_index in chunk_indices {
        log_info(format!(
            "Running diagnostic training-chunk validation chunk {} phase {:?}",
            chunk_index, phase
        ));
        run_chunk_validation::<M>(args, chunk_index, phase).await;
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
        chunk_index,
        all_chunks,
        phase,
        login_smoke,
    } = CliArgs::parse();
    let config_contents = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read config file '{}': {}", config_path, err));
    let args: Args = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));
    set_title(&format!(
        "training_chunk_validation_{}_{}",
        args.model_cli_name, args.config_nickname_training
    ));
    check_sympy_availability().unwrap();
    let model_name = LlmModelName::from_str(&args.model_cli_name, true).unwrap();
    assert!(
        chunk_index.is_none() || !all_chunks,
        "pass either --chunk-index or --all-chunks, not both"
    );
    assert!(
        all_chunks || chunk_index.is_some(),
        "no chunk selector was provided; use --all-chunks or --chunk-index"
    );
    if login_smoke {
        println!(
            "login-smoke passed for bin_oneshot_training_chunk_validation: model={}, training_config={}, generation_config={}, phase={:?}, chunk={}",
            args.model_cli_name,
            args.config_nickname_training,
            args.config_nickname_generation,
            phase,
            chunk_index
                .map(|chunk_index| chunk_index.to_string())
                .unwrap_or_else(|| "all".to_string())
        );
        return;
    }
    let summary_log = format!(
        "{}/small_files/{}/{}/training_chunk_validation/{}_summary.txt",
        args.mount_dir,
        args.model_cli_name,
        args.config_nickname_training,
        chunk_index
            .map(|chunk_index| format!("chunk_{chunk_index}"))
            .unwrap_or_else(|| "all_chunks".to_string())
    );
    let verbose_log = format!(
        "{}/small_files/{}/{}/training_chunk_validation/{}_verbose.txt",
        args.mount_dir,
        args.model_cli_name,
        args.config_nickname_training,
        chunk_index
            .map(|chunk_index| format!("chunk_{chunk_index}"))
            .unwrap_or_else(|| "all_chunks".to_string())
    );
    ensure_parent_dir_exists(&summary_log).unwrap();
    ProgressTextLogger::initialize(summary_log, verbose_log)
        .await
        .unwrap();
    match model_name {
        LlmModelName::Gemma3_4b => {
            run_selected_chunk_validations::<Gemma3_4BIt>(&args, chunk_index, all_chunks, phase)
                .await
        }
        LlmModelName::Llama31_8b => {
            run_selected_chunk_validations::<Llama31_8BInstruct>(
                &args,
                chunk_index,
                all_chunks,
                phase,
            )
            .await
        }
        LlmModelName::Mistral7bInstructV03 => {
            run_selected_chunk_validations::<Mistral7BInstructV03>(
                &args,
                chunk_index,
                all_chunks,
                phase,
            )
            .await
        }
        LlmModelName::Qwen3_06b => {
            run_selected_chunk_validations::<Qwen3_06B>(&args, chunk_index, all_chunks, phase).await
        }
        LlmModelName::Qwen3_4b => {
            run_selected_chunk_validations::<Qwen3_4B>(&args, chunk_index, all_chunks, phase).await
        }
        LlmModelName::Qwen25_7b => {
            run_selected_chunk_validations::<Qwen25_7B>(&args, chunk_index, all_chunks, phase).await
        }
        LlmModelName::Qwen35_08b => {
            run_selected_chunk_validations::<Qwen35_08B>(&args, chunk_index, all_chunks, phase)
                .await
        }
        LlmModelName::Qwen35_4b => {
            run_selected_chunk_validations::<Qwen35_4B>(&args, chunk_index, all_chunks, phase).await
        }
    }
    ProgressTextLogger::shutdown().await.unwrap();
}
