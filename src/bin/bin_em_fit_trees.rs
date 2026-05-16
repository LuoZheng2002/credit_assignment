use clap::Parser;
use credit_assignment::asset_file::AssetFile;
use credit_assignment::em::em_schema::AssetFileEmFit;
use credit_assignment::em::em_types::{EmHyperparameters, LogStdClamp};
use credit_assignment::llm_model::LlmModel;

#[derive(Parser, Debug)]
#[command(name = "Fit EM over trees")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,

    #[arg(short, long)]
    num_samples: usize,

    #[arg(value_enum, short, long)]
    model: LlmModel,

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
    let em_fit_file = AssetFileEmFit {
        model: args.model,
        dataset: args.dataset_name,
        num_samples: args.num_samples,
        hyperparameters: EmHyperparameters {
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
        },
    };
    let (per_tree, _meta) = em_fit_file.fetch();
    let written_count = per_tree.len();
    let output_file = em_fit_file.per_tree_file_path();
    let em_fit_meta_file = em_fit_file.meta_file_path();
    println!(
        "Wrote {} EmFitPerTree entries to {}",
        written_count, output_file
    );
    println!("Wrote EM fit metadata to {}", em_fit_meta_file);
}
