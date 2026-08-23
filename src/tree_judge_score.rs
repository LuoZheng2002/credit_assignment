use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    chunked_judging::{
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT, DEFAULT_CACHE_VERSION,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL, JudgingSummary, judge_requests,
        read_judging_outputs,
    },
    get_accuracy::{
        AccuracyStats, DatasetAccuracies, TestAccuracyResult, equal_dataset_macro_average,
        get_accuracy_from_tree_judgments_at_path, get_test_accuracies_from_tree_judgments_at_path,
    },
    hybrid_dataset::{DatasetSplit, Testing},
    llm_model::LlmModelMarker,
    tree_artifact::{
        TreeArtifact, TreeJudgment, read_available_tree_artifact_chunks,
        read_marked_tree_artifact_chunks,
    },
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, clap::ValueEnum)]
pub enum TreeArtifactReadMode {
    Available,
    Marked,
}

pub fn read_tree_artifacts_for_judging<M, S>(
    tree_artifact_path: &str,
    read_mode: TreeArtifactReadMode,
) -> Result<Vec<TreeArtifact<M, S>>, String>
where
    M: LlmModelMarker,
    S: DatasetSplit,
{
    match read_mode {
        TreeArtifactReadMode::Available => {
            read_available_tree_artifact_chunks::<M, S>(tree_artifact_path)
        }
        TreeArtifactReadMode::Marked => {
            read_marked_tree_artifact_chunks::<M, S>(tree_artifact_path)
        }
    }
}

pub async fn judge_tree_artifacts_at_path<M, S>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    judging_output_jsonl_path: &str,
    cache_dir: &str,
    escalation_jsonl_path: &str,
    read_mode: TreeArtifactReadMode,
) -> Result<JudgingSummary, String>
where
    M: LlmModelMarker,
    S: DatasetSplit,
{
    let artifacts = read_tree_artifacts_for_judging::<M, S>(tree_artifact_path, read_mode)?;
    let requests = artifacts
        .iter()
        .flat_map(|artifact| artifact.to_judging_requests(DEFAULT_CACHE_VERSION))
        .collect::<Vec<_>>();
    let summary = judge_requests(
        requests,
        Path::new(judging_output_jsonl_path),
        Path::new(cache_dir),
        Path::new(escalation_jsonl_path),
        DEFAULT_CACHE_VERSION,
        DEFAULT_CACHE_CHUNK_QUESTION_COUNT,
        DEFAULT_REQUEST_CONCURRENCY_PER_MODEL,
    )
    .await?;
    write_tree_judgments_from_outputs::<M, S>(
        artifacts,
        judging_output_jsonl_path,
        tree_judgment_jsonl_path,
    )?;
    Ok(summary)
}

fn write_tree_judgments_from_outputs<M, S>(
    artifacts: Vec<TreeArtifact<M, S>>,
    judging_output_jsonl_path: &str,
    tree_judgment_jsonl_path: &str,
) -> Result<(), String>
where
    M: LlmModelMarker,
    S: DatasetSplit,
{
    let outputs = read_judging_outputs(Path::new(judging_output_jsonl_path))?;
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
    if let Some(parent) = Path::new(tree_judgment_jsonl_path).parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create tree judgment parent {}: {err}",
                parent.display()
            )
        })?;
    }
    let mut file = File::create(tree_judgment_jsonl_path).map_err(|err| {
        format!(
            "failed to create tree judgment JSONL {}: {err}",
            tree_judgment_jsonl_path
        )
    })?;
    for artifact in artifacts {
        let outputs = outputs_by_artifact_id
            .remove(&artifact.artifact_id)
            .unwrap_or_default();
        let judgment = TreeJudgment::from_judging_outputs(
            artifact.artifact_id.clone(),
            DEFAULT_CACHE_VERSION.to_string(),
            S::dataset_file_postfix(),
            artifact.question.flat_id.0,
            outputs,
        )?;
        serde_json::to_writer(&mut file, &judgment).map_err(|err| {
            format!(
                "failed to serialize tree judgment to {}: {err}",
                tree_judgment_jsonl_path
            )
        })?;
        writeln!(file).map_err(|err| {
            format!(
                "failed to write tree judgment JSONL {}: {err}",
                tree_judgment_jsonl_path
            )
        })?;
    }
    Ok(())
}

pub async fn score_tree_judgments_at_path<M, S>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    progress_bar_label: &str,
) -> AccuracyStats
where
    M: LlmModelMarker,
    S: DatasetSplit,
{
    get_accuracy_from_tree_judgments_at_path::<M, S>(
        tree_artifact_path,
        tree_judgment_jsonl_path,
        progress_bar_label,
    )
    .await
}

pub async fn score_testing_tree_judgments_at_path<M>(
    tree_artifact_path: &str,
    tree_judgment_jsonl_path: &str,
    progress_bar_label: &str,
    num_trunks: usize,
) -> TestAccuracyResult
where
    M: LlmModelMarker,
{
    get_test_accuracies_from_tree_judgments_at_path::<M, Testing>(
        tree_artifact_path,
        tree_judgment_jsonl_path,
        progress_bar_label,
        num_trunks,
    )
    .await
}

pub fn trial_tree_artifact_path(base_path: &str, trial_index: usize) -> String {
    format!("{base_path}/trial_{trial_index}")
}

pub fn trial_tree_judgment_path(base_path: &str, trial_index: usize) -> String {
    let path = Path::new(base_path);
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("tree_judgments_oneshot");
    let extension = path.extension().and_then(|name| name.to_str());
    match extension {
        Some(extension) => parent
            .join(format!("{stem}_trial_{trial_index}.{extension}"))
            .to_string_lossy()
            .to_string(),
        None => parent
            .join(format!("{stem}_trial_{trial_index}"))
            .to_string_lossy()
            .to_string(),
    }
}

pub fn testing_trial_tree_artifact_path(base_path: &str, trial_index: usize) -> String {
    trial_tree_artifact_path(base_path, trial_index)
}

pub fn testing_trial_tree_judgment_path(base_path: &str, trial_index: usize) -> String {
    trial_tree_judgment_path(base_path, trial_index)
}

fn summarize_accuracy_values(accuracy_values: Vec<f32>) -> DatasetAccuracies {
    let n = accuracy_values.len() as f32;
    let mean = if n > 0.0 {
        accuracy_values.iter().sum::<f32>() / n
    } else {
        0.0
    };
    let variance = if n > 1.0 {
        accuracy_values
            .iter()
            .map(|accuracy| (accuracy - mean).powi(2))
            .sum::<f32>()
            / (n - 1.0)
    } else {
        0.0
    };
    let std_err = if n > 0.0 { (variance / n).sqrt() } else { 0.0 };
    DatasetAccuracies {
        accuracy_values,
        mean_accuracy: mean,
        confidence_interval_half_width: 1.96 * std_err,
    }
}

pub fn merge_test_accuracy_results(results: Vec<TestAccuracyResult>) -> TestAccuracyResult {
    let mut per_dataset = BTreeMap::new();
    for result in results {
        for (dataset_name, dataset_accuracies) in result.per_dataset {
            per_dataset
                .entry(dataset_name)
                .or_insert_with(Vec::new)
                .extend(dataset_accuracies.accuracy_values);
        }
    }
    let per_dataset = per_dataset
        .into_iter()
        .map(|(dataset_name, accuracy_values)| {
            (dataset_name, summarize_accuracy_values(accuracy_values))
        })
        .collect::<BTreeMap<_, _>>();
    let macro_average = equal_dataset_macro_average(&per_dataset);
    TestAccuracyResult {
        per_dataset,
        macro_average,
    }
}
