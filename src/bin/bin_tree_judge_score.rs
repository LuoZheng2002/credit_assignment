use std::{
    backtrace::Backtrace,
    path::{Path, PathBuf},
};

use clap::{Parser, ValueEnum};
use credit_assignment::{
    check_python_env::check_sympy_availability,
    chunked_judging::build_judging_metadata,
    hybrid_dataset::{DatasetSplitEnum, Testing, Training, Validation},
    json_toml_utils::write_json,
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelMarker, LlmModelName, Mistral7BInstructV03,
        Qwen3_4B, Qwen3_06B, Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    progress_text_logger::ProgressTextLogger,
    tree_judge_score::{
        TreeArtifactReadMode, judge_tree_artifacts_at_path, merge_test_accuracy_results,
        score_testing_tree_judgments_at_path, score_tree_judgments_at_path,
        testing_trial_tree_artifact_path, testing_trial_tree_judgment_path,
    },
};
use proctitle::set_title;

#[derive(Parser, Debug)]
#[command(about = "Judge and score direct tree artifacts independent of rollout binaries")]
struct CliArgs {
    #[arg(long)]
    model_cli_name: String,
    #[arg(long)]
    dataset_split: DatasetSplitEnum,
    #[arg(long, default_value = "judge-score")]
    phase: JudgeScorePhase,
    #[arg(long)]
    tree_artifact_path: String,
    #[arg(long)]
    tree_judgment_jsonl_path: String,
    #[arg(long)]
    judging_output_jsonl_path: Option<String>,
    #[arg(long)]
    cache_dir: Option<String>,
    #[arg(long)]
    escalation_jsonl: Option<String>,
    #[arg(long)]
    score_output_json: Option<PathBuf>,
    #[arg(long)]
    metadata_json: Option<PathBuf>,
    #[arg(long)]
    log_summary_path: Option<PathBuf>,
    #[arg(long)]
    log_verbose_path: Option<PathBuf>,
    #[arg(long, default_value = "marked")]
    read_mode: TreeArtifactReadMode,
    #[arg(long)]
    num_rollout_trials: Option<usize>,
    #[arg(long)]
    num_trunks: Option<usize>,
    #[arg(long, default_value = "Tree artifact accuracy")]
    progress_label: String,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
}

fn default_metadata_path(args: &CliArgs) -> PathBuf {
    if let Some(path) = &args.metadata_json {
        return path.clone();
    }
    if matches!(
        args.phase,
        JudgeScorePhase::Judge | JudgeScorePhase::JudgeScore
    ) {
        if let Some(path) = &args.judging_output_jsonl_path {
            return PathBuf::from(path).with_extension("metadata.json");
        }
    }
    if let Some(path) = &args.score_output_json {
        return path.with_extension("metadata.json");
    }
    PathBuf::from(&args.tree_judgment_jsonl_path).with_extension("metadata.json")
}

fn default_log_paths(args: &CliArgs) -> (PathBuf, PathBuf) {
    let metadata_path = default_metadata_path(args);
    let parent = metadata_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = metadata_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("tree_judge_score");
    (
        args.log_summary_path
            .clone()
            .unwrap_or_else(|| parent.join(format!("{stem}_summary.txt"))),
        args.log_verbose_path
            .clone()
            .unwrap_or_else(|| parent.join(format!("{stem}_verbose.txt"))),
    )
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum JudgeScorePhase {
    Judge,
    Score,
    JudgeScore,
}

macro_rules! dispatch_split {
    ($model_name:expr, $split:expr, $func:ident, $($args:expr),* $(,)?) => {{
        match $split {
            DatasetSplitEnum::Training => match $model_name {
                LlmModelName::Qwen25_7b => $func::<Qwen25_7B, Training>($($args),*).await,
                LlmModelName::Qwen3_06b => $func::<Qwen3_06B, Training>($($args),*).await,
                LlmModelName::Qwen3_4b => $func::<Qwen3_4B, Training>($($args),*).await,
                LlmModelName::Qwen35_4b => $func::<Qwen35_4B, Training>($($args),*).await,
                LlmModelName::Qwen35_08b => $func::<Qwen35_08B, Training>($($args),*).await,
                LlmModelName::Gemma3_4b => $func::<Gemma3_4BIt, Training>($($args),*).await,
                LlmModelName::Llama31_8b => $func::<Llama31_8BInstruct, Training>($($args),*).await,
                LlmModelName::Mistral7bInstructV03 => $func::<Mistral7BInstructV03, Training>($($args),*).await,
            },
            DatasetSplitEnum::Validation => match $model_name {
                LlmModelName::Qwen25_7b => $func::<Qwen25_7B, Validation>($($args),*).await,
                LlmModelName::Qwen3_06b => $func::<Qwen3_06B, Validation>($($args),*).await,
                LlmModelName::Qwen3_4b => $func::<Qwen3_4B, Validation>($($args),*).await,
                LlmModelName::Qwen35_4b => $func::<Qwen35_4B, Validation>($($args),*).await,
                LlmModelName::Qwen35_08b => $func::<Qwen35_08B, Validation>($($args),*).await,
                LlmModelName::Gemma3_4b => $func::<Gemma3_4BIt, Validation>($($args),*).await,
                LlmModelName::Llama31_8b => $func::<Llama31_8BInstruct, Validation>($($args),*).await,
                LlmModelName::Mistral7bInstructV03 => $func::<Mistral7BInstructV03, Validation>($($args),*).await,
            },
            DatasetSplitEnum::Testing => match $model_name {
                LlmModelName::Qwen25_7b => $func::<Qwen25_7B, Testing>($($args),*).await,
                LlmModelName::Qwen3_06b => $func::<Qwen3_06B, Testing>($($args),*).await,
                LlmModelName::Qwen3_4b => $func::<Qwen3_4B, Testing>($($args),*).await,
                LlmModelName::Qwen35_4b => $func::<Qwen35_4B, Testing>($($args),*).await,
                LlmModelName::Qwen35_08b => $func::<Qwen35_08B, Testing>($($args),*).await,
                LlmModelName::Gemma3_4b => $func::<Gemma3_4BIt, Testing>($($args),*).await,
                LlmModelName::Llama31_8b => $func::<Llama31_8BInstruct, Testing>($($args),*).await,
                LlmModelName::Mistral7bInstructV03 => $func::<Mistral7BInstructV03, Testing>($($args),*).await,
            },
        }
    }};
}

async fn judge_single<M, S>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    judging_output_jsonl_path: &str,
    metadata_path: &Path,
    cache_dir: &str,
    escalation_jsonl: &str,
    read_mode: TreeArtifactReadMode,
) -> Result<(), String>
where
    M: LlmModelMarker,
    S: credit_assignment::hybrid_dataset::DatasetSplit,
{
    let summary = judge_tree_artifacts_at_path::<M, S>(
        tree_artifact_path,
        tree_judgment_jsonl_path,
        judging_output_jsonl_path,
        cache_dir,
        escalation_jsonl,
        read_mode,
    )
    .await?;
    println!("Judging summary: {summary:#?}");
    let metadata = build_judging_metadata(
        "tree_judge_score_judge",
        PathBuf::from(judging_output_jsonl_path).as_path(),
        summary,
    )?;
    write_json(metadata_path.to_string_lossy().as_ref(), &metadata)?;
    println!("Judging metadata written to {}", metadata_path.display());
    Ok(())
}

async fn score_single<M, S>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    progress_label: &str,
    score_output_json: Option<&PathBuf>,
) -> Result<(), String>
where
    M: LlmModelMarker,
    S: credit_assignment::hybrid_dataset::DatasetSplit,
{
    let score = score_tree_judgments_at_path::<M, S>(
        tree_artifact_path,
        tree_judgment_jsonl_path,
        progress_label,
    )
    .await;
    println!("Score: {score:#?}");
    if let Some(path) = score_output_json {
        write_json(path.to_string_lossy().as_ref(), &score)?;
        println!("Score written to {}", path.display());
        let metadata_path = path.with_extension("metadata.json");
        let metadata = serde_json::json!({
            "schema_version": 1,
            "stage": "tree_judge_score_score",
            "tree_artifact_path": tree_artifact_path,
            "tree_judgment_jsonl_path": tree_judgment_jsonl_path,
            "score_output_json": path,
            "score": score,
        });
        write_json(metadata_path.to_string_lossy().as_ref(), &metadata)?;
        println!("Score metadata written to {}", metadata_path.display());
    }
    Ok(())
}

async fn judge_testing<M: LlmModelMarker>(args: &CliArgs) -> Result<(), String> {
    if let Some(num_rollout_trials) = args.num_rollout_trials {
        for trial_index in 0..num_rollout_trials {
            let tree_artifact_path =
                testing_trial_tree_artifact_path(&args.tree_artifact_path, trial_index);
            let tree_judgment_jsonl_path =
                testing_trial_tree_judgment_path(&args.tree_judgment_jsonl_path, trial_index);
            let judging_output_jsonl_path = testing_trial_tree_judgment_path(
                args.judging_output_jsonl_path
                    .as_deref()
                    .expect("--judging-output-jsonl is required for judge phases"),
                trial_index,
            );
            let metadata_path =
                PathBuf::from(&judging_output_jsonl_path).with_extension("metadata.json");
            judge_single::<M, Testing>(
                &tree_artifact_path,
                &tree_judgment_jsonl_path,
                &judging_output_jsonl_path,
                &metadata_path,
                args.cache_dir
                    .as_deref()
                    .expect("--cache-dir is required for judge phases"),
                args.escalation_jsonl
                    .as_deref()
                    .expect("--escalation-jsonl is required for judge phases"),
                args.read_mode,
            )
            .await?;
        }
    } else {
        judge_single::<M, Testing>(
            &args.tree_artifact_path,
            &args.tree_judgment_jsonl_path,
            args.judging_output_jsonl_path
                .as_deref()
                .expect("--judging-output-jsonl is required for judge phases"),
            &default_metadata_path(args),
            args.cache_dir
                .as_deref()
                .expect("--cache-dir is required for judge phases"),
            args.escalation_jsonl
                .as_deref()
                .expect("--escalation-jsonl is required for judge phases"),
            args.read_mode,
        )
        .await?;
    }
    Ok(())
}

async fn score_testing<M: LlmModelMarker>(args: &CliArgs) -> Result<(), String> {
    let num_trunks = args
        .num_trunks
        .expect("--num-trunks is required when scoring testing artifacts");
    let score = if let Some(num_rollout_trials) = args.num_rollout_trials {
        let mut results = Vec::new();
        for trial_index in 0..num_rollout_trials {
            results.push(
                score_testing_tree_judgments_at_path::<M>(
                    &testing_trial_tree_artifact_path(&args.tree_artifact_path, trial_index),
                    &testing_trial_tree_judgment_path(&args.tree_judgment_jsonl_path, trial_index),
                    &args.progress_label,
                    num_trunks,
                )
                .await,
            );
        }
        merge_test_accuracy_results(results)
    } else {
        score_testing_tree_judgments_at_path::<M>(
            &args.tree_artifact_path,
            &args.tree_judgment_jsonl_path,
            &args.progress_label,
            num_trunks,
        )
        .await
    };
    println!("Testing score: {score:#?}");
    if let Some(path) = &args.score_output_json {
        write_json(path.to_string_lossy().as_ref(), &score)?;
        println!("Testing score written to {}", path.display());
        let metadata_path = path.with_extension("metadata.json");
        let metadata = serde_json::json!({
            "schema_version": 1,
            "stage": "tree_judge_score_testing_score",
            "tree_artifact_path": args.tree_artifact_path,
            "tree_judgment_jsonl_path": args.tree_judgment_jsonl_path,
            "score_output_json": path,
            "num_rollout_trials": args.num_rollout_trials,
            "num_trunks": args.num_trunks,
            "score": score,
        });
        write_json(metadata_path.to_string_lossy().as_ref(), &metadata)?;
        println!(
            "Testing score metadata written to {}",
            metadata_path.display()
        );
    }
    Ok(())
}

async fn run_non_testing<M, S>(args: &CliArgs) -> Result<(), String>
where
    M: LlmModelMarker,
    S: credit_assignment::hybrid_dataset::DatasetSplit,
{
    if matches!(
        args.phase,
        JudgeScorePhase::Judge | JudgeScorePhase::JudgeScore
    ) {
        judge_single::<M, S>(
            &args.tree_artifact_path,
            &args.tree_judgment_jsonl_path,
            args.judging_output_jsonl_path
                .as_deref()
                .expect("--judging-output-jsonl is required for judge phases"),
            &default_metadata_path(args),
            args.cache_dir
                .as_deref()
                .expect("--cache-dir is required for judge phases"),
            args.escalation_jsonl
                .as_deref()
                .expect("--escalation-jsonl is required for judge phases"),
            args.read_mode,
        )
        .await?;
    }
    if matches!(
        args.phase,
        JudgeScorePhase::Score | JudgeScorePhase::JudgeScore
    ) {
        score_single::<M, S>(
            &args.tree_artifact_path,
            &args.tree_judgment_jsonl_path,
            &args.progress_label,
            args.score_output_json.as_ref(),
        )
        .await?;
    }
    Ok(())
}

async fn run_testing<M: LlmModelMarker>(args: &CliArgs) -> Result<(), String> {
    if matches!(
        args.phase,
        JudgeScorePhase::Judge | JudgeScorePhase::JudgeScore
    ) {
        judge_testing::<M>(args).await?;
    }
    if matches!(
        args.phase,
        JudgeScorePhase::Score | JudgeScorePhase::JudgeScore
    ) {
        score_testing::<M>(args).await?;
    }
    Ok(())
}

macro_rules! dispatch_testing {
    ($model_name:expr, $args:expr) => {{
        match $model_name {
            LlmModelName::Qwen25_7b => run_testing::<Qwen25_7B>($args).await,
            LlmModelName::Qwen3_06b => run_testing::<Qwen3_06B>($args).await,
            LlmModelName::Qwen3_4b => run_testing::<Qwen3_4B>($args).await,
            LlmModelName::Qwen35_4b => run_testing::<Qwen35_4B>($args).await,
            LlmModelName::Qwen35_08b => run_testing::<Qwen35_08B>($args).await,
            LlmModelName::Gemma3_4b => run_testing::<Gemma3_4BIt>($args).await,
            LlmModelName::Llama31_8b => run_testing::<Llama31_8BInstruct>($args).await,
            LlmModelName::Mistral7bInstructV03 => run_testing::<Mistral7BInstructV03>($args).await,
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
    set_title("tree_judge_score");
    check_sympy_availability().unwrap();
    let args = CliArgs::parse();
    let model_name = LlmModelName::from_str(&args.model_cli_name, true)
        .unwrap_or_else(|err| panic!("failed to parse --model-cli-name: {err}"));
    if args.login_smoke {
        println!(
            "login-smoke passed for bin_tree_judge_score: model={}, split={:?}, phase={:?}",
            args.model_cli_name, args.dataset_split, args.phase
        );
        return;
    }
    let (log_summary_path, log_verbose_path) = default_log_paths(&args);
    ProgressTextLogger::initialize(log_summary_path, log_verbose_path)
        .await
        .unwrap_or_else(|err| panic!("failed to initialize progress logger: {err}"));
    let result = match args.dataset_split {
        DatasetSplitEnum::Testing => dispatch_testing!(model_name, &args),
        DatasetSplitEnum::Training | DatasetSplitEnum::Validation => {
            dispatch_split!(model_name, args.dataset_split, run_non_testing, &args)
        }
    };
    if let Err(err) = ProgressTextLogger::shutdown()
        .await
        .map_err(|err| err.to_string())
    {
        if result.is_ok() {
            panic!("failed to shutdown progress logger: {err}");
        }
        eprintln!("warning: progress logger shutdown failed after error: {err}");
    }
    result.unwrap_or_else(|err| panic!("{err}"));
}
