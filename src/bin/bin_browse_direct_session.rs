use std::backtrace::Backtrace;

use clap::{Parser};
use credit_assignment::{direct_tool::{direct_tree_action_log::AssetFileDirectTreeActionLogs, posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters, TemperatureAccuracyPair}}, json_line_util::read_json, llm_model::LlmModelName};
use research_utility::asset_file::AssetFile;

// home page: view the questions and win rate
// the questions should be paged; each page should have 10 questions

// tree page: after clicking a question, we should enter the tree page. It should be of vertical layout.
// The top is a summary window with question, correct answer, accuracy and an optional model answer if we click on a leaf segment.
// The middle is a conversation window that shows the conversation up to the segment the user clicks on
// The bottom is the tree view like the one in src/bin/bin_browse_session.rs, but now it shows the segments instead of nodes
// The left and right arrow controls how many actions are considered to build the tree, it should demonstrate how the tree evolves with more actions applied
// We can click on a segment in the tree to show the conversation up to that segment in the conversation window;
// if a leaf segment is clicked, we can also show the model answer and the correctness judgment in the summary window.

// use the key q to transition from tree page to home page, and press q again to exit the program

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactively browse rollout session logs")]
struct Args {
    #[arg(value_enum, short, long)]
    model: LlmModelName,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long)]
    temperature_to_accuracy_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
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
    let Args {
        model,
        config_nickname,
        rollout_config_path,
        temperature_to_accuracy_path,
        posterior_hyperparameters_path,
    } = Args::parse();
    let rollout_config = read_json(rollout_config_path).unwrap();
    let temperature_to_accuracy =
        read_json::<Vec<TemperatureAccuracyPair>>(temperature_to_accuracy_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        temperature_to_accuracy,
        hyperparameters: posterior_hyperparameters,
    };
    let asset_file_action_logs = AssetFileDirectTreeActionLogs {
        model,
        nickname: config_nickname,
        rollout_config,
        posterior_calculation_config,
    };
    let action_log_store = asset_file_action_logs.fetch().await;
    let mut keys = action_log_store.get_keys().await.unwrap();
    keys.sort();
    // to do: implement the two pages and the navigation logic
}
