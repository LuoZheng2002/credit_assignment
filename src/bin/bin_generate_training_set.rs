use clap::Parser;
use credit_assignment::llm_model::LlmModel;
use credit_assignment::em::em_types::{EmHyperparameters, LogStdClamp};
use credit_assignment::training_set::training_set_batch::AssetFileTrainingBatch;
use credit_assignment::asset_file::AssetFile;

const DEFAULT_BATCH_SIZE: usize = 4;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate tokenized training batches")]
struct Args {
    #[arg(value_enum, short, long)]
    model: LlmModel,

    #[arg(short, long)]
    dataset_name: String,

    #[arg(short, long, alias = "num_steps")]
    num_samples: usize,

    #[arg(long, default_value_t = 1.0)]
    sigma_ordinary: f64,

    #[arg(long, default_value_t = 1.0)]
    sigma_special: f64,

    #[arg(long, default_value_t = 1.0)]
    sigma_log_std: f64,

    #[arg(long, default_value_t = 1.0)]
    lambda_slack: f64,

    #[arg(long, default_value_t = 1e-6)]
    eps: f64,

    #[arg(long, default_value_t = 100)]
    max_iterations: usize,

    #[arg(long, default_value_t = -4.0)]
    log_std_min: f64,

    #[arg(long, default_value_t = 2.0)]
    log_std_max: f64,
}

fn main() {
    let args = Args::parse();
    let hyperparameters = EmHyperparameters {
        sigma_ordinary: args.sigma_ordinary,
        sigma_special: args.sigma_special,
        sigma_log_std: args.sigma_log_std,
        lambda_slack: args.lambda_slack,
        eps: args.eps,
        max_iterations: args.max_iterations,
        log_std_clamp: LogStdClamp {
            min: args.log_std_min,
            max: args.log_std_max,
        },
    };

    let training_batch_asset = AssetFileTrainingBatch {
        model: args.model,
        dataset: args.dataset_name,
        num_samples: args.num_samples,
        hyperparameters,
        batch_size: DEFAULT_BATCH_SIZE,
    };
    training_batch_asset.synchronize();
    println!(
        "Generated training batches at {}",
        training_batch_asset.file_path()
    );
}
