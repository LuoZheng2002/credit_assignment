use std::backtrace::Backtrace;

use clap::{Parser, ValueEnum};
use credit_assignment::{
    direct_tool::{
        direct_rollout_config::DirectRolloutConfig,
        direct_tree::DirectTree,
        direct_tree_action_log::{AssetFileDirectTreeActionLogs, DirectTreeActionLog},
        posterior_calculation_config::{PosteriorCalculationConfig, PosteriorHyperparameters},
    },
    json_line_util::read_json,
    llm_model::{Gpt4o, LlmModelMarker, LlmModelName, Qwen3_4B, Qwen25, Qwen35_4B, Qwen35_08B},
};
use research_utility::asset_file::AssetFile;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Get the accuracy of a model under a temperature from direct rollout logs"
)]
struct Args {
    #[arg(long)]
    model_cli_name: String,
    #[arg(long)]
    config_nickname: String,
    #[arg(long)]
    rollout_config_path: String,
    #[arg(long)]
    posterior_hyperparameters_path: String,
}

pub fn question_is_correct<M: LlmModelMarker>(action_log: &DirectTreeActionLog<M>) -> bool {
    let tree = DirectTree::<M>::from_action_log(action_log);
    assert!(
        tree.leaf_segment_judgments.len() == 1,
        "There should be exactly one leaf segment judgment for accuracy calculation"
    );
    let judgment = tree.leaf_segment_judgments.values().next().unwrap();
    judgment.is_correct
}

async fn print_accuracy_for_model<M: LlmModelMarker>(
    config_nickname: String,
    rollout_config: DirectRolloutConfig,
    posterior_calculation_config: PosteriorCalculationConfig,
) {
    let asset_file_action_logs = AssetFileDirectTreeActionLogs::<M> {
        nickname: config_nickname,
        rollout_config,
        posterior_calculation_config,
        _phantom: std::marker::PhantomData,
    };
    let action_log_store = asset_file_action_logs.fetch().await;
    let mut keys = action_log_store.get_keys().await.unwrap();
    keys.sort();
    let mut total = 0;
    let mut correct = 0;
    let mut incorrect = 0;
    for key in keys {
        let action_log = action_log_store
            .get(key)
            .await
            .unwrap()
            .expect("key from sqlite key set must exist");
        let is_correct = question_is_correct::<M>(&action_log);
        total += 1;
        if is_correct {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
    assert_eq!(total, correct + incorrect);
    let accuracy = correct as f64 / total as f64;
    println!("Total questions: {}", total);
    println!("Correct questions: {}", correct);
    println!("Incorrect questions: {}", incorrect);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
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
        model_cli_name,
        config_nickname,
        rollout_config_path,
        posterior_hyperparameters_path,
    } = Args::parse();
    let rollout_config: DirectRolloutConfig = read_json(rollout_config_path).unwrap();
    let posterior_hyperparameters =
        read_json::<PosteriorHyperparameters>(posterior_hyperparameters_path).unwrap();
    let posterior_calculation_config = PosteriorCalculationConfig {
        hyperparameters: posterior_hyperparameters,
    };
    let model_name = LlmModelName::from_str(&model_cli_name, true).unwrap();
    match model_name {
        LlmModelName::Gpt4o => {
            print_accuracy_for_model::<Gpt4o>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
            )
            .await;
        }
        LlmModelName::Qwen25_7b => {
            print_accuracy_for_model::<Qwen25>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
            )
            .await;
        }
        LlmModelName::Qwen3_4b => {
            print_accuracy_for_model::<Qwen3_4B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
            )
            .await;
        }
        LlmModelName::Qwen35_08b => {
            print_accuracy_for_model::<Qwen35_08B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
            )
            .await;
        }
        LlmModelName::Qwen35_4b => {
            print_accuracy_for_model::<Qwen35_4B>(
                config_nickname,
                rollout_config,
                posterior_calculation_config,
            )
            .await;
        }
    }
}
