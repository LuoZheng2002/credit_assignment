use clap::Parser;
use credit_assignment::direct_answer::generate_raw_answers::LlmModel;
use credit_assignment::em::em_schema::AssetFileEmFit;
use credit_assignment::version_tracking::AssetFile;

#[derive(Parser, Debug)]
#[command(name = "Fit EM over trees")]
struct Args {
    #[arg(short, long)]
    dataset_name: String,

    #[arg(short, long)]
    num_samples: usize,

    #[arg(value_enum, short, long)]
    model: LlmModel,
}

fn main() {
    let args = Args::parse();
    let em_fit_file = AssetFileEmFit {
        model: args.model,
        dataset: args.dataset_name,
        num_samples: args.num_samples,
    };
    let (per_tree, _meta) = em_fit_file.fetch();
    let written_count = per_tree.len();
    let output_file = em_fit_file.per_tree_file_path();
    let em_fit_meta_file = em_fit_file.meta_file_path();
    println!(
        "Wrote {} EmFitPerTree entries to {}",
        written_count,
        output_file
    );
    println!(
        "Wrote EM fit metadata to {}",
        em_fit_meta_file
    );
}
