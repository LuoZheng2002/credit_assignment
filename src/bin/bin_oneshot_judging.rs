use std::{backtrace::Backtrace, collections::BTreeMap, fs::File, io::Write, path::PathBuf};

use clap::{Parser, ValueEnum};
use credit_assignment::{
    chunked_judging::{
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT, DEFAULT_CACHE_VERSION,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL, JudgingRequestRecord, judge_requests,
        read_judging_outputs, read_judging_requests,
    },
    hybrid_dataset::{DatasetSplitEnum, Testing, Training, Validation},
    llm_model::{
        Gemma3_4BIt, Llama31_8BInstruct, LlmModelName, Mistral7BInstructV03, Qwen3_4B, Qwen3_06B,
        Qwen25_7B, Qwen35_4B, Qwen35_08B,
    },
    tree_artifact::{TreeJudgment, read_marked_tree_artifact_chunks},
    utils::write_json,
};
use proctitle::set_title;

#[derive(Parser, Debug)]
#[command(about = "Judge model answers with the chunked OpenRouter judgment cache")]
struct CliArgs {
    #[arg(long, conflicts_with = "input_tree_msgpack")]
    input_jsonl: Option<PathBuf>,
    #[arg(long, conflicts_with = "input_jsonl")]
    input_tree_msgpack: Option<PathBuf>,
    #[arg(long)]
    output_jsonl: PathBuf,
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long)]
    escalation_jsonl: PathBuf,
    #[arg(long, default_value = DEFAULT_CACHE_VERSION)]
    cache_version: String,
    #[arg(long, default_value_t = DEFAULT_CACHE_CHUNK_QUESTION_COUNT)]
    cache_chunk_question_count: usize,
    #[arg(long, default_value_t = DEFAULT_REQUEST_CONCURRENCY_PER_MODEL)]
    request_concurrency_per_model: usize,
    #[arg(long)]
    summary_json: Option<PathBuf>,
    #[arg(long, requires = "input_tree_msgpack")]
    output_tree_judgment_jsonl: Option<PathBuf>,
    #[arg(long, requires = "input_tree_msgpack")]
    model_cli_name: Option<String>,
    #[arg(long, requires = "input_tree_msgpack")]
    dataset_split: Option<DatasetSplitEnum>,
    #[arg(long, default_value_t = false)]
    login_smoke: bool,
}

fn requests_from_tree_artifacts<M, S>(
    tree_msgpack_path: &PathBuf,
    cache_version: &str,
) -> Result<Vec<JudgingRequestRecord>, String>
where
    M: credit_assignment::llm_model::LlmModelMarker,
    S: credit_assignment::hybrid_dataset::DatasetSplit,
{
    let artifacts = read_marked_tree_artifact_chunks::<M, S>(tree_msgpack_path)?;
    Ok(artifacts
        .iter()
        .flat_map(|artifact| artifact.to_judging_requests(cache_version))
        .collect())
}

fn write_tree_judgments<M, S>(
    tree_msgpack_path: &PathBuf,
    judging_output_jsonl_path: &PathBuf,
    output_tree_judgment_jsonl_path: &PathBuf,
    cache_version: &str,
) -> Result<(), String>
where
    M: credit_assignment::llm_model::LlmModelMarker,
    S: credit_assignment::hybrid_dataset::DatasetSplit,
{
    let artifacts = read_marked_tree_artifact_chunks::<M, S>(tree_msgpack_path)?;
    let outputs = read_judging_outputs(judging_output_jsonl_path)?;
    let mut outputs_by_artifact_id = BTreeMap::<String, Vec<_>>::new();
    for output in outputs {
        let Some(artifact_id) = output.request.artifact_id.clone() else {
            continue;
        };
        outputs_by_artifact_id
            .entry(artifact_id)
            .or_default()
            .push(output);
    }
    if let Some(parent) = output_tree_judgment_jsonl_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create tree judgment directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let mut file = File::create(output_tree_judgment_jsonl_path).map_err(|err| {
        format!(
            "failed to create tree judgment output {}: {err}",
            output_tree_judgment_jsonl_path.display()
        )
    })?;
    for artifact in artifacts {
        let outputs = outputs_by_artifact_id
            .remove(&artifact.artifact_id)
            .unwrap_or_default();
        let judgment = TreeJudgment::from_judging_outputs(
            artifact.artifact_id.clone(),
            cache_version.to_string(),
            S::dataset_file_postfix(),
            artifact.question.flat_id.0,
            outputs,
        )?;
        let _checked = credit_assignment::tree_artifact::TreeJudged::<M, S>::new(
            artifact.clone(),
            judgment.clone(),
        )?;
        serde_json::to_writer(&mut file, &judgment).map_err(|err| {
            format!(
                "failed to serialize tree judgment to {}: {err}",
                output_tree_judgment_jsonl_path.display()
            )
        })?;
        writeln!(file).map_err(|err| {
            format!(
                "failed to write tree judgment to {}: {err}",
                output_tree_judgment_jsonl_path.display()
            )
        })?;
    }
    Ok(())
}

macro_rules! dispatch_model_split {
    ($model_name:expr, $dataset_split:expr, $func:ident, $($args:expr),* $(,)?) => {{
        match $dataset_split {
            DatasetSplitEnum::Training => match $model_name {
                LlmModelName::Qwen25_7b => $func::<Qwen25_7B, Training>($($args),*),
                LlmModelName::Qwen3_06b => $func::<Qwen3_06B, Training>($($args),*),
                LlmModelName::Qwen3_4b => $func::<Qwen3_4B, Training>($($args),*),
                LlmModelName::Qwen35_4b => $func::<Qwen35_4B, Training>($($args),*),
                LlmModelName::Qwen35_08b => $func::<Qwen35_08B, Training>($($args),*),
                LlmModelName::Gemma3_4b => $func::<Gemma3_4BIt, Training>($($args),*),
                LlmModelName::Llama31_8b => $func::<Llama31_8BInstruct, Training>($($args),*),
                LlmModelName::Mistral7bInstructV03 => $func::<Mistral7BInstructV03, Training>($($args),*),
            },
            DatasetSplitEnum::Validation => match $model_name {
                LlmModelName::Qwen25_7b => $func::<Qwen25_7B, Validation>($($args),*),
                LlmModelName::Qwen3_06b => $func::<Qwen3_06B, Validation>($($args),*),
                LlmModelName::Qwen3_4b => $func::<Qwen3_4B, Validation>($($args),*),
                LlmModelName::Qwen35_4b => $func::<Qwen35_4B, Validation>($($args),*),
                LlmModelName::Qwen35_08b => $func::<Qwen35_08B, Validation>($($args),*),
                LlmModelName::Gemma3_4b => $func::<Gemma3_4BIt, Validation>($($args),*),
                LlmModelName::Llama31_8b => $func::<Llama31_8BInstruct, Validation>($($args),*),
                LlmModelName::Mistral7bInstructV03 => $func::<Mistral7BInstructV03, Validation>($($args),*),
            },
            DatasetSplitEnum::Testing => match $model_name {
                LlmModelName::Qwen25_7b => $func::<Qwen25_7B, Testing>($($args),*),
                LlmModelName::Qwen3_06b => $func::<Qwen3_06B, Testing>($($args),*),
                LlmModelName::Qwen3_4b => $func::<Qwen3_4B, Testing>($($args),*),
                LlmModelName::Qwen35_4b => $func::<Qwen35_4B, Testing>($($args),*),
                LlmModelName::Qwen35_08b => $func::<Qwen35_08B, Testing>($($args),*),
                LlmModelName::Gemma3_4b => $func::<Gemma3_4BIt, Testing>($($args),*),
                LlmModelName::Llama31_8b => $func::<Llama31_8BInstruct, Testing>($($args),*),
                LlmModelName::Mistral7bInstructV03 => $func::<Mistral7BInstructV03, Testing>($($args),*),
            },
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
    set_title("oneshot_judging");
    let args = CliArgs::parse();
    assert!(
        args.cache_chunk_question_count > 0,
        "--cache-chunk-question-count must be positive"
    );
    assert!(
        args.request_concurrency_per_model > 0,
        "--request-concurrency-per-model must be positive"
    );
    match (&args.input_jsonl, &args.input_tree_msgpack) {
        (Some(_), None) | (None, Some(_)) => {}
        _ => panic!("provide exactly one of --input-jsonl or --input-tree-msgpack"),
    }
    if let Some(model_cli_name) = &args.model_cli_name {
        LlmModelName::from_str(model_cli_name, true)
            .unwrap_or_else(|err| panic!("failed to parse --model-cli-name: {err}"));
    }
    if args.input_tree_msgpack.is_some() {
        assert!(
            args.model_cli_name.is_some(),
            "--model-cli-name is required with --input-tree-msgpack"
        );
        assert!(
            args.dataset_split.is_some(),
            "--dataset-split is required with --input-tree-msgpack"
        );
    }
    if args.login_smoke {
        println!(
            "login-smoke passed for bin_oneshot_judging: input_jsonl={}, input_tree_msgpack={}, output_jsonl={}, cache_dir={}",
            args.input_jsonl.is_some(),
            args.input_tree_msgpack.is_some(),
            args.output_jsonl.display(),
            args.cache_dir.display()
        );
        return;
    }

    let requests = match (&args.input_jsonl, &args.input_tree_msgpack) {
        (Some(input_jsonl), None) => {
            read_judging_requests(input_jsonl).unwrap_or_else(|err| panic!("{err}"))
        }
        (None, Some(input_tree_msgpack)) => {
            let model_cli_name = args
                .model_cli_name
                .as_ref()
                .expect("--model-cli-name is required with --input-tree-msgpack");
            let model_name = LlmModelName::from_str(model_cli_name, true)
                .unwrap_or_else(|err| panic!("failed to parse --model-cli-name: {err}"));
            let dataset_split = args
                .dataset_split
                .expect("--dataset-split is required with --input-tree-msgpack");
            dispatch_model_split!(
                model_name,
                dataset_split,
                requests_from_tree_artifacts,
                input_tree_msgpack,
                &args.cache_version,
            )
            .unwrap_or_else(|err| panic!("{err}"))
        }
        _ => panic!("provide exactly one of --input-jsonl or --input-tree-msgpack"),
    };

    let summary = judge_requests(
        requests,
        &args.output_jsonl,
        &args.cache_dir,
        &args.escalation_jsonl,
        &args.cache_version,
        args.cache_chunk_question_count,
        args.request_concurrency_per_model,
    )
    .await
    .unwrap_or_else(|err| {
        eprintln!("judging_failed=1 reason={err}");
        panic!("{err}")
    });

    if let Some(output_tree_judgment_jsonl) = &args.output_tree_judgment_jsonl {
        let input_tree_msgpack = args
            .input_tree_msgpack
            .as_ref()
            .expect("--output-tree-judgment-jsonl requires --input-tree-msgpack");
        let model_cli_name = args
            .model_cli_name
            .as_ref()
            .expect("--model-cli-name is required with --output-tree-judgment-jsonl");
        let model_name = LlmModelName::from_str(model_cli_name, true)
            .unwrap_or_else(|err| panic!("failed to parse --model-cli-name: {err}"));
        let dataset_split = args
            .dataset_split
            .expect("--dataset-split is required with --output-tree-judgment-jsonl");
        dispatch_model_split!(
            model_name,
            dataset_split,
            write_tree_judgments,
            input_tree_msgpack,
            &args.output_jsonl,
            output_tree_judgment_jsonl,
            &args.cache_version,
        )
        .unwrap_or_else(|err| panic!("{err}"));
    }

    println!("Judging summary: {summary:#?}");
    let summary_path = args
        .summary_json
        .unwrap_or_else(|| args.output_jsonl.with_extension("summary.json"));
    write_json(summary_path.to_string_lossy().as_ref(), &summary)
        .unwrap_or_else(|err| panic!("failed to write summary: {err}"));
    println!("Judging summary written to {}", summary_path.display());
}
