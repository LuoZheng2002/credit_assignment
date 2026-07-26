use std::backtrace::Backtrace;

use clap::{Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    constants,
    hybrid_dataset::{DatasetSplit, DatasetSplitEnum, Testing, Training, Validation},
    json_toml_utils::read_json,
    llm_model::{
        Gemma3_4BIt, InferenceEndpoint, Llama31_8BInstruct, LlmModelMarker, LlmModelName,
        Mistral7BInstructV03, Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    rollout::{RolloutProgramConfig, rollout_all},
    rollout_config::RolloutConfig,
    tree_to_action::BranchingRuntimeOptions,
};
use reqwest::Client;
use research_utility::progress_text_logger::ProgressTextLogger;

const DEFAULT_PROGRESS_TEXT_LOG_PATH: &str = "progress_log";

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run direct tree rollout and save action logs"
)]
struct Args {
    #[arg(long)]
    model_cli_name: String,
    #[arg(long, conflicts_with = "sglang_base_url")]
    sglang_port: Option<u16>,
    #[arg(long, conflicts_with = "sglang_port")]
    sglang_base_url: Option<String>,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long)]
    use_tool: bool,
    #[arg(long, value_enum)]
    dataset_split: DatasetSplitEnum,
    #[arg(long)]
    epoch: usize, // the epoch index
    #[arg(long)]
    total_epochs: usize,
    #[arg(long)]
    rollout_secs: usize,
    #[arg(long)]
    inference_wrapper_log_path: String,
    #[arg(long, default_value = DEFAULT_PROGRESS_TEXT_LOG_PATH)]
    progress_text_log_path: String,
}

async fn run_rollout_for_split<M: LlmModelMarker, S: DatasetSplit>(
    rollout_config: RolloutConfig<S>,
    args: &Args,
    client: Client,
    posterior_calculation_config: PosteriorCalculationConfig,
    inference_endpoint: InferenceEndpoint,
) {
    let program_config = RolloutProgramConfig {
        config_nickname: args.config_nickname.clone(),
        rollout_config,
        posterior_calculation_config,
        epoch: args.epoch,
        client,
        inference_endpoint,
        rollout_secs: args.rollout_secs,
        total_epochs: args.total_epochs,
        action_log_store_override_path: None,
        use_tool: args.use_tool,
        fixed_temperature: constants::temperature_by_split::<S>(),
        max_concurrent_rollout: 300,
        branching_options: BranchingRuntimeOptions::default(),
    };
    let _ = rollout_all::<M, S>("results", program_config).await;
}

macro_rules! run_rollout {
    (
        $model_name:expr,
        $dataset_split:expr,
        $args:expr,
        $client:expr,
        $posterior:expr,
        $inference_endpoint:expr;
        $( $model_enum:path, $model_ty:ty ),+ $(,)?;
        $( $split_enum:path, $split_ty:ty ),+ $(,)?
    ) => {{
        let model_name = $model_name;
        let dataset_split = $dataset_split;
        let args = $args;
        let client = $client;
        let posterior = $posterior;
        let inference_endpoint = $inference_endpoint;

        macro_rules! run_model_for_split {
            ($rollout_config:expr, $inner_split_ty:ty, $endpoint:expr) => {
                match model_name {
                    $(
                        $model_enum => {
                            run_rollout_for_split::<$model_ty, $inner_split_ty>(
                                $rollout_config,
                                args,
                                client,
                                posterior,
                                $endpoint,
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
                    run_model_for_split!(rollout_config, $split_ty, inference_endpoint.clone())
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
    let args = Args::parse();
    check_sympy_availability().unwrap();
    assert!(args.total_epochs > 0, "total_epochs must be positive");

    println!("Starting direct rollout evaluation pipeline...");
    let client = Client::new();
    let posterior_hyperparameters = read_json::<PosteriorHyperparameters>(
        credit_assignment::directories::POSTERIOR_HYPERPARAMETERS_PATH,
    )
    .unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };

    let model_name = LlmModelName::from_str(&args.model_cli_name, true).unwrap();
    let inference_endpoint =
        InferenceEndpoint::from_cli_options(args.sglang_port, args.sglang_base_url.clone())
            .unwrap();
    ProgressTextLogger::initialize(
        format!("{}_summary.txt", args.progress_text_log_path),
        format!("{}_verbose.txt", args.progress_text_log_path),
    )
    .await
    .unwrap();
    run_rollout!(
        model_name,
        args.dataset_split,
        &args,
        client,
        posterior_calculation_config,
        inference_endpoint;
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
