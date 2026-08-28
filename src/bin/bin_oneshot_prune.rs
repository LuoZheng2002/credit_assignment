use std::backtrace::Backtrace;

use clap::{Parser, ValueEnum};
use credit_assignment::{
    directories::{oneshot_epochs_parent_dir, training_summary_oneshot_parent_dir},
    launch_inference_wrapper::InferenceBackend,
    llm_model::LlmModelName,
    oneshot_training_summary::read_existing_validation_summary,
    python_training_config::TrainingHyperparameters,
    utils::configure_mount_dir,
};
use proctitle::set_title;
use research_utility::progress_text_logger::{ProgressTextLogger, log_info};
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(about = "Prune non-best one-shot checkpoints after validation jobs complete")]
struct CliArgs {
    #[arg(short = 'c', long)]
    config_path: String,
    #[arg(long, default_value_t = 10)]
    epoch_interval: usize,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
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

#[tokio::main(flavor = "current_thread")]
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
        epoch_interval,
        login_smoke,
    } = CliArgs::parse();
    assert!(epoch_interval > 0, "--epoch-interval must be positive");
    let config_contents = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read config file '{}': {}", config_path, err));
    let args: Args = toml::from_str(&config_contents)
        .unwrap_or_else(|err| panic!("failed to parse config file '{}': {}", config_path, err));
    let validation_total_epochs = args
        .validation_total_epochs
        .unwrap_or(args.num_oneshot_epochs);
    assert!(
        validation_total_epochs > 0,
        "validation_total_epochs/num_oneshot_epochs must be positive"
    );
    LlmModelName::from_str(&args.model_cli_name, true)
        .unwrap_or_else(|err| panic!("invalid model_cli_name: {}", err));
    set_title(&format!(
        "oneshot_prune_{}_{}",
        args.model_cli_name, args.config_nickname_training
    ));
    if login_smoke {
        println!(
            "login-smoke passed for bin_oneshot_prune: model={}, training_config={}, validation_epochs={}, epoch_interval={}",
            args.model_cli_name,
            args.config_nickname_training,
            validation_total_epochs,
            epoch_interval
        );
        return;
    }
    configure_mount_dir(&args.mount_dir)
        .unwrap_or_else(|err| panic!("failed to configure mount dir: {}", err));

    let summary_parent_dir = training_summary_oneshot_parent_dir(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_training,
    );
    let model_output_root = oneshot_epochs_parent_dir(
        &args.mount_dir,
        &args.model_cli_name,
        &args.config_nickname_training,
    );
    ProgressTextLogger::initialize(
        format!("{summary_parent_dir}/prune_summary.txt"),
        format!("{summary_parent_dir}/prune_verbose.txt"),
    )
    .await
    .unwrap();

    let (validated_epochs, _validation_accuracies) =
        read_existing_validation_summary(&summary_parent_dir);
    let expected_validated_epochs = (0..=validation_total_epochs)
        .filter(|epoch| *epoch == 0 || *epoch % epoch_interval == 0)
        .collect::<Vec<_>>();
    let missing_epochs = expected_validated_epochs
        .iter()
        .copied()
        .filter(|epoch| !validated_epochs.contains(epoch))
        .collect::<Vec<_>>();
    assert!(
        missing_epochs.is_empty(),
        "cannot prune before interval held-out validation is complete; epoch_interval={}, missing expected validated epochs: {:?}",
        epoch_interval,
        missing_epochs
    );
    let candidate_epochs = (1..=validation_total_epochs)
        .filter(|epoch| epoch % epoch_interval != 0)
        .collect::<Vec<_>>();
    log_info(format!(
        "Pruning only non-validation-interval checkpoints after interval held-out validation: epoch_interval={}, expected_validated_epochs={:?}, candidate_epochs={:?}",
        epoch_interval, expected_validated_epochs, candidate_epochs
    ));
    for epoch in candidate_epochs {
        let epoch_dir =
            std::path::Path::new(&model_output_root).join(format!("oneshot_epoch_{epoch}"));
        if !epoch_dir.exists() {
            continue;
        }
        match std::fs::remove_dir_all(&epoch_dir) {
            Ok(()) => log_info(format!(
                "Pruned non-validation-interval oneshot model snapshot at {}",
                epoch_dir.display()
            )),
            Err(err) => log_info(format!(
                "Failed to prune non-validation-interval oneshot model snapshot at {}: {}",
                epoch_dir.display(),
                err
            )),
        }
    }
    ProgressTextLogger::shutdown().await.unwrap();
}
