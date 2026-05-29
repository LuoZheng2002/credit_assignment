use std::backtrace::Backtrace;

use clap::Parser;
use credit_assignment::{
    direct_tool::{
        direct_rollout_config::{AdvantageCalculationPolicy, DirectRolloutConfig},
        direct_training_set::AssetFileTrainingTrajectories,
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{Gpt4o, LlmModelMarker, LlmModelName, Qwen3_4B, Qwen25_7B, Qwen35_4B, Qwen35_08B},
};
use research_utility::asset_file::AssetFile;
#[derive(Parser, Debug)]
#[command(author, version, about = "Interactively browse training sets")]
struct Args {
    #[arg(value_enum, short, long)]
    model: LlmModelName,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
    #[arg(long)]
    epoch: usize, // the epoch index
    #[arg(long)]
    max_num_training_trajectories: usize,
    #[arg(long, value_enum)]
    advantage_calculation_policy: AdvantageCalculationPolicy,
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
    let Args {
        model,
        config_nickname,
        rollout_config_path,
        posterior_hyperparameters_path,
        epoch,
        max_num_training_trajectories,
        advantage_calculation_policy,
    } = Args::parse();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let run_program_args = RunProgramArgs {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        max_num_training_trajectories,
        advantage_calculation_policy,
    };
    match model {
        LlmModelName::Gpt4o => run_program::<Gpt4o>(run_program_args).await,
        LlmModelName::Qwen3_4b => run_program::<Qwen3_4B>(run_program_args).await,
        LlmModelName::Qwen25_7b => run_program::<Qwen25_7B>(run_program_args).await,
        LlmModelName::Qwen35_08b => run_program::<Qwen35_08B>(run_program_args).await,
        LlmModelName::Qwen35_4b => run_program::<Qwen35_4B>(run_program_args).await,
    }
}

struct RunProgramArgs {
    config_nickname: String,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
    epoch: usize, // the epoch index
    max_num_training_trajectories: usize,
    advantage_calculation_policy: AdvantageCalculationPolicy,
}

async fn run_program<M: LlmModelMarker>(run_program_args: RunProgramArgs) {
    let RunProgramArgs {
        config_nickname,
        rollout_config,
        posterior_calculation_config,
        epoch,
        max_num_training_trajectories,
        advantage_calculation_policy,
    } = run_program_args;
    let asset_file_training_set = AssetFileTrainingTrajectories::<M> {
        config_nickname: config_nickname.clone(),
        rollout_config,
        posterior_calculation_config,
        epoch,
        max_num_training_trajectories,
        advantage_calculation_policy,
        _phantom: std::marker::PhantomData::<M>,
    };
    asset_file_training_set.synchronize().await;
}
