use std::backtrace::Backtrace;

use clap::{Parser, ValueEnum};
use proctitle::set_title;
use serde::Deserialize;

use credit_assignment::{
    check_python_env::check_sympy_availability,
    directories::{
        text_logger_summary_path, text_logger_verbose_path, training_trajectories_oneshot_path,
        training_trajectories_stats_oneshot_path, tree_artifacts_oneshot_path,
        tree_judgments_oneshot_path,
    },
    hybrid_dataset::Training,
    json_toml_utils::read_json,
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelName, Mistral7BInstructV03, Qwen3_4B, Qwen3_06B,
        Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    rollout_config::{RolloutConfig, TrainingAdvantagePolicy},
    training_set::{TrainingSetSortMode, tree_judgments_to_training_trajectories},
    utils::configure_mount_dir,
};
use research_utility::progress_text_logger::ProgressTextLogger;

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(short = 'c', long)]
    config_path: String,
    #[arg(long)]
    positive_advantage_only: Option<bool>,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Args {
    model_cli_name: String,
    config_nickname_rollout: String,
    config_nickname_generation: String,
    rollout_config_path: String,
    use_tool: bool,
    #[serde(default)]
    epoch: usize,
    training_advantage_policy: TrainingAdvantagePolicy,
    positive_advantage_only: bool,
    rollout_mount_dir: String,
    generation_mount_dir: String,
    num_gpus: usize,
    #[serde(default)]
    total_time_limit_hours: f32,
    training_set_sort_mode: TrainingSetSortMode,
    #[serde(default)]
    num_questions_per_chunk: Option<usize>,
    #[serde(default)]
    num_chunks: Option<usize>,
}

macro_rules! generate_trajectories {
    (
        $model_name:expr,
        $tree_artifacts_msgpack_path:expr,
        $tree_judgment_jsonl_path:expr,
        $trajectories_dir:expr,
        $trajectories_msgpack_path:expr,
        $stats_path:expr,
        $config_bundle_path:expr,
        $rollout_config:expr,
        $posterior_calculation_config:expr,
        $training_advantage_policy:expr,
        $positive_advantage_only:expr,
        $use_tool:expr,
        $training_set_sort_mode:expr,
        $num_questions_per_chunk:expr,
        $num_chunks:expr;
        $( $model_enum:path, $model_ty:ty ),+ $(,)?
    ) => {
        match $model_name {
            $(
                $model_enum => {
                    std::fs::create_dir_all($trajectories_dir).unwrap();
                    if std::path::Path::new($trajectories_msgpack_path).exists() {
                        std::fs::remove_file($trajectories_msgpack_path).unwrap();
                    }
                    if let Ok(entries) = std::fs::read_dir($trajectories_dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            let is_stale_chunk = path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .is_some_and(|name| {
                                    name.starts_with("chunk_") && name.ends_with(".msgpack")
                                });
                            if is_stale_chunk {
                                std::fs::remove_file(&path).unwrap_or_else(|err| {
                                    panic!(
                                        "failed to remove stale trajectory chunk {}: {}",
                                        path.display(),
                                        err
                                    )
                                });
                            }
                        }
                    }
                    credit_assignment::json_toml_utils::write_json(
                        $config_bundle_path,
                        &credit_assignment::training_set::TrainingTrajectoryConfigBundle {
                            rollout_config: $rollout_config.clone(),
                            posterior_calculation_config: $posterior_calculation_config.clone(),
                        },
                    )
                    .unwrap();
                    tree_judgments_to_training_trajectories::<$model_ty>(
                        $tree_artifacts_msgpack_path,
                        $tree_judgment_jsonl_path,
                        $trajectories_msgpack_path.to_string(),
                        $stats_path.to_string(),
                        $rollout_config,
                        $posterior_calculation_config,
                        $training_advantage_policy,
                        $positive_advantage_only,
                        $use_tool,
                        $training_set_sort_mode,
                        $num_questions_per_chunk,
                        $num_chunks,
                    )
                    .await
                }
            ),+
        }
    };
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
        positive_advantage_only,
        login_smoke,
    } = CliArgs::parse();
    let config_contents = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read config file '{}': {}", config_path, err));
    let mut args: Args = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));
    if let Some(positive_advantage_only) = positive_advantage_only {
        args.positive_advantage_only = positive_advantage_only;
    }
    let process_title = format!(
        "oneshot_generation_{}_{}",
        args.model_cli_name, args.config_nickname_generation
    );
    set_title(&process_title);
    check_sympy_availability().unwrap();
    assert!(args.num_gpus > 0, "--num-gpus must be positive");
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )
    .unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let rollout_config: RolloutConfig<Training> = read_json(&args.rollout_config_path).unwrap();
    let model_name = LlmModelName::from_str(&args.model_cli_name, true).unwrap();

    let trajectories_dir = training_trajectories_oneshot_path(
        &args.generation_mount_dir,
        &args.model_cli_name,
        &args.config_nickname_generation,
    );
    let trajectories_msgpack_path = format!("{}/trajectories.msgpack", trajectories_dir);
    let stats_path = training_trajectories_stats_oneshot_path(
        &args.generation_mount_dir,
        &args.model_cli_name,
        &args.config_nickname_generation,
    );
    let config_bundle_path = format!("{}/config_bundle.json", trajectories_dir);
    let tree_artifacts_msgpack_path = tree_artifacts_oneshot_path::<Training>(
        &args.rollout_mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
        args.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to resolve tree artifacts path: {}", err));
    let tree_judgment_jsonl_path = tree_judgments_oneshot_path::<Training>(
        &args.rollout_mount_dir,
        &args.model_cli_name,
        &args.config_nickname_rollout,
        args.epoch,
    )
    .unwrap_or_else(|err| panic!("failed to resolve tree judgments path: {}", err));
    if login_smoke {
        println!(
            "login-smoke passed for bin_oneshot_generation: model={}, rollout_config={}, generation_config={}, tree_artifacts={}, tree_judgments={}",
            args.model_cli_name,
            args.config_nickname_rollout,
            args.config_nickname_generation,
            tree_artifacts_msgpack_path,
            tree_judgment_jsonl_path
        );
        return;
    }
    configure_mount_dir(&args.rollout_mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure rollout mount dir: {}", err));
    configure_mount_dir(&args.generation_mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure generation mount dir: {}", err));

    let text_log_summary_path = text_logger_summary_path(
        &args.generation_mount_dir,
        &args.model_cli_name,
        &args.config_nickname_generation,
    );
    let text_log_verbose_path = text_logger_verbose_path(
        &args.generation_mount_dir,
        &args.model_cli_name,
        &args.config_nickname_generation,
    );
    ProgressTextLogger::initialize(text_log_summary_path, text_log_verbose_path)
        .await
        .unwrap();

    generate_trajectories!(
        model_name,
        &tree_artifacts_msgpack_path,
        &tree_judgment_jsonl_path,
        &trajectories_dir,
        &trajectories_msgpack_path,
        &stats_path,
        &config_bundle_path,
        rollout_config,
        posterior_calculation_config,
        args.training_advantage_policy,
        args.positive_advantage_only,
        args.use_tool,
        args.training_set_sort_mode,
        args.num_questions_per_chunk,
        args.num_chunks;
        LlmModelName::Qwen25_7b, Qwen25_7B,
        LlmModelName::Qwen3_06b, Qwen3_06B,
        LlmModelName::Qwen3_4b, Qwen3_4B,
        LlmModelName::Qwen35_4b, Qwen35_4B,
        LlmModelName::Qwen35_08b, Qwen35_08B,
        LlmModelName::Gemma3_4b, Gemma3_4BIt,
        LlmModelName::Llama31_8b, Llama31_8BInstruct,
        LlmModelName::Mistral7bInstructV03, Mistral7BInstructV03
    );
    println!(
        "Training trajectories generated at {}",
        trajectories_msgpack_path
    );

    ProgressTextLogger::shutdown().await.unwrap();
}
