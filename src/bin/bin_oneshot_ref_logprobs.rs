use std::backtrace::Backtrace;
use std::path::Path;
use std::process::Command;

use clap::Parser;
use proctitle::set_title;
use serde::Deserialize;

use credit_assignment::{
    check_python_env::check_sympy_availability,
    directories::{
        base_model_dir, text_logger_summary_path, text_logger_verbose_path,
        training_trajectories_oneshot_path,
    },
    launch_inference_wrapper::InferenceBackend,
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelMarker, Mistral7BInstructV03, Qwen3_4B,
        Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    python_training_config::TrainingHyperparameters,
    utils::configure_mount_dir,
};
use research_utility::progress_text_logger::{log_info, log_state, ProgressTextLogger};

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(short = 'c', long)]
    config_path: String,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: usize,
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
    training_set_sort_mode: Option<String>,
    #[serde(default)]
    total_time_limit_hours: f32,
    mount_dir: String,
    generation_mount_dir: String,
}

async fn run_ref_logprobs<M: LlmModelMarker>(args: &Args, cli: &CliArgs) {
    let trajectories_dir = training_trajectories_oneshot_path(
        &args.generation_mount_dir,
        M::CLI_NAME,
        &args.config_nickname_generation,
    );
    if !Path::new(&trajectories_dir).is_dir() {
        panic!(
            "Training trajectory directory not found at {}; run bin_oneshot_generation first",
            trajectories_dir
        );
    }
    let base_model_parent = base_model_dir(&args.mount_dir, M::CLI_NAME);
    let nested_model_path = Path::new(&base_model_parent)
        .join("model")
        .to_string_lossy()
        .into_owned();
    let model_path = if Path::new(&nested_model_path).join("config.json").exists() {
        nested_model_path
    } else {
        base_model_parent
    };
    let mut command = Command::new("python3");
    command
        .arg("-m")
        .arg("src_py.ref_logprobs.annotate")
        .arg("--input-dir")
        .arg(&trajectories_dir)
        .arg("--model-path-or-name")
        .arg(&model_path)
        .arg("--fallback-model-name")
        .arg(M::API_NAME)
        .arg("--max-batch-tokens")
        .arg(cli.max_batch_tokens.to_string());
    if cli.login_smoke {
        command.arg("--login-smoke");
    }
    log_info(format!(
        "Running reference-logprob annotation for {} trajectories_dir={} model_path={} fallback_model={}",
        M::CLI_NAME,
        trajectories_dir,
        model_path,
        M::API_NAME
    ));
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to launch reference-logprob annotator: {}", err));
    if !status.success() {
        panic!("reference-logprob annotator exited with status {}", status);
    }
}

macro_rules! dispatch_model {
    ($model_cli_name:expr, $args:expr, $cli:expr, $($model:ty),+ $(,)?) => {{
        match $model_cli_name {
            $(<$model>::CLI_NAME => run_ref_logprobs::<$model>($args, $cli).await,)+
            other => panic!("Unsupported model_cli_name for bin_oneshot_ref_logprobs: {}", other),
        }
    }};
}

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("{}", panic_info);
        eprintln!("Backtrace:\n{}", Backtrace::force_capture());
    }));
    set_title("oneshot_ref_logprobs");
    check_sympy_availability().unwrap();

    let cli = CliArgs::parse();
    let config_contents = std::fs::read_to_string(&cli.config_path)
        .unwrap_or_else(|err| panic!("failed to read config {}: {}", cli.config_path, err));
    let args: Args = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config {}: {}", cli.config_path, err));

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
    configure_mount_dir(&args.generation_mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure generation mount dir: {}", err));
    configure_mount_dir(&args.mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));

    if cli.login_smoke {
        log_info(format!(
            "login-smoke passed for bin_oneshot_ref_logprobs: model={}, generation_config={}, kl_beta={}, max_batch_tokens={}",
            args.model_cli_name,
            args.config_nickname_generation,
            args.training_hyperparameters.kl_beta,
            cli.max_batch_tokens
        ));
    }

    dispatch_model!(
        args.model_cli_name.as_str(),
        &args,
        &cli,
        Qwen25_7B,
        Qwen3_4B,
        Qwen3_06B,
        Qwen35_4B,
        Qwen35_08B,
        Gemma3_4BIt,
        Mistral7BInstructV03,
        Llama31_8BInstruct,
    );
    log_state("Reference-logprob annotation completed");
    ProgressTextLogger::shutdown().await.unwrap();
}
