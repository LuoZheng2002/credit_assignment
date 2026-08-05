use std::{collections::BTreeMap, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    chunked_judging::{JudgingOutputRecord, JudgingRequestRecord, judgment_cache_key},
    directories::tree_artifacts_oneshot_chunk_path,
    hybrid_dataset::{DatasetSplit, HybridDatasetQuestion},
    judge_correctness::CorrectnessJudgment,
    llm_model::LlmModelMarker,
    trajectory::FinalAnswer,
    tree::{DirectTree, Segment, SegmentId},
};

pub const DIRECT_TREE_ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeLeafAnswer {
    pub leaf_segment_id: SegmentId,
    pub final_answer: FinalAnswer,
    pub model_answer_string: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TreeArtifact<M: LlmModelMarker, S: DatasetSplit> {
    pub schema_version: u32,
    pub artifact_id: String,
    pub model_cli_name: String,
    pub config_nickname: String,
    pub epoch: usize,
    pub use_tool: bool,
    pub question: HybridDatasetQuestion<S>,
    pub root_segment_id: SegmentId,
    pub trunk_leaf_segments: Vec<SegmentId>,
    pub segments: Vec<Segment<M>>,
    pub leaf_answers: Vec<TreeLeafAnswer>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TreeJudgment {
    pub schema_version: u32,
    pub artifact_id: String,
    pub cache_version: String,
    pub split: String,
    pub flat_id: usize,
    pub judgments_by_model_answer: BTreeMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TreeJudged<M: LlmModelMarker, S: DatasetSplit> {
    pub tree: TreeArtifact<M, S>,
    pub judgment: TreeJudgment,
}

impl<M: LlmModelMarker, S: DatasetSplit> TreeArtifact<M, S> {
    pub fn from_direct_tree(
        tree: &DirectTree<M, S>,
        artifact_id: String,
        model_cli_name: String,
        config_nickname: String,
        epoch: usize,
    ) -> Self {
        let leaf_answers = tree
            .leaf_segment_answers
            .iter()
            .map(|(leaf_segment_id, final_answer)| TreeLeafAnswer {
                leaf_segment_id: *leaf_segment_id,
                final_answer: final_answer.clone(),
                model_answer_string: final_answer.model_answer_text().to_string(),
            })
            .collect::<Vec<_>>();
        Self {
            schema_version: DIRECT_TREE_ARTIFACT_SCHEMA_VERSION,
            artifact_id,
            model_cli_name,
            config_nickname,
            epoch,
            use_tool: tree.action_log.use_tool,
            question: tree.action_log.question.clone(),
            root_segment_id: tree.root_segment_id.expect("direct tree root must exist"),
            trunk_leaf_segments: tree.trunk_leaf_segments.iter().copied().collect(),
            segments: tree.segments.values().cloned().collect(),
            leaf_answers,
        }
    }

    pub fn to_judging_requests(&self, cache_version: &str) -> Vec<JudgingRequestRecord> {
        self.leaf_answers
            .iter()
            .map(|leaf| JudgingRequestRecord {
                artifact_id: Some(self.artifact_id.clone()),
                split: S::dataset_file_postfix(),
                flat_id: self.question.flat_id.0,
                question: self.question.question.clone(),
                correct_answer: self.question.correct_answer.clone(),
                model_answer: leaf.model_answer_string.clone(),
                metadata: serde_json::json!({
                    "schema_version": DIRECT_TREE_ARTIFACT_SCHEMA_VERSION,
                    "leaf_segment_id": leaf.leaf_segment_id,
                    "model_cli_name": self.model_cli_name,
                    "config_nickname": self.config_nickname,
                    "epoch": self.epoch,
                    "cache_key": judgment_cache_key(
                        cache_version,
                        &S::dataset_file_postfix(),
                        self.question.flat_id.0,
                        &leaf.model_answer_string,
                    ),
                }),
            })
            .collect()
    }
}

impl TreeJudgment {
    pub fn from_judging_outputs(
        artifact_id: String,
        cache_version: String,
        split: String,
        flat_id: usize,
        outputs: impl IntoIterator<Item = JudgingOutputRecord>,
    ) -> Result<Self, String> {
        let mut judgments_by_model_answer = BTreeMap::new();
        for output in outputs {
            if output.request.artifact_id.as_deref() != Some(artifact_id.as_str()) {
                continue;
            }
            if output.request.split != split || output.request.flat_id != flat_id {
                return Err(format!(
                    "judging output for artifact {} has mismatched split/flat_id: {}/{} vs {}/{}",
                    artifact_id, output.request.split, output.request.flat_id, split, flat_id
                ));
            }
            if let Some(previous) = judgments_by_model_answer
                .insert(output.request.model_answer.clone(), output.is_correct)
            {
                if previous != output.is_correct {
                    return Err(format!(
                        "conflicting judgments for artifact {} answer {:?}: {} vs {}",
                        artifact_id, output.request.model_answer, previous, output.is_correct
                    ));
                }
            }
        }
        Ok(Self {
            schema_version: DIRECT_TREE_ARTIFACT_SCHEMA_VERSION,
            artifact_id,
            cache_version,
            split,
            flat_id,
            judgments_by_model_answer,
        })
    }

    pub fn get_correctness_for_answer(&self, model_answer_string: &str) -> Option<bool> {
        self.judgments_by_model_answer
            .get(model_answer_string)
            .copied()
    }
}

impl<M: LlmModelMarker, S: DatasetSplit> TreeJudged<M, S> {
    pub fn new(tree: TreeArtifact<M, S>, judgment: TreeJudgment) -> Result<Self, String> {
        if tree.artifact_id != judgment.artifact_id {
            return Err(format!(
                "tree artifact id {} does not match judgment artifact id {}",
                tree.artifact_id, judgment.artifact_id
            ));
        }
        if tree.question.flat_id.0 != judgment.flat_id {
            return Err(format!(
                "tree flat_id {} does not match judgment flat_id {}",
                tree.question.flat_id.0, judgment.flat_id
            ));
        }
        for leaf in &tree.leaf_answers {
            if !judgment
                .judgments_by_model_answer
                .contains_key(&leaf.model_answer_string)
            {
                return Err(format!(
                    "tree artifact {} leaf {:?} answer {:?} has no judgment",
                    tree.artifact_id, leaf.leaf_segment_id, leaf.model_answer_string
                ));
            }
        }
        Ok(Self { tree, judgment })
    }

    pub fn legacy_leaf_judgments(&self) -> BTreeMap<SegmentId, CorrectnessJudgment> {
        self.tree
            .leaf_answers
            .iter()
            .map(|leaf| {
                let is_correct = self
                    .judgment
                    .judgments_by_model_answer
                    .get(&leaf.model_answer_string)
                    .copied()
                    .expect("TreeJudged must have one verdict for every leaf answer");
                (
                    leaf.leaf_segment_id,
                    CorrectnessJudgment {
                        model_answer: leaf.final_answer.clone(),
                        correct_answer: self.tree.question.correct_answer.clone(),
                        is_correct,
                    },
                )
            })
            .collect()
    }
}

pub fn write_tree_artifacts_msgpack<M: LlmModelMarker, S: DatasetSplit>(
    path: impl AsRef<Path>,
    artifacts: &[TreeArtifact<M, S>],
) -> Result<(), String> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create tree artifact directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let bytes = rmp_serde::to_vec(artifacts)
        .map_err(|err| format!("failed to serialize tree artifacts: {err}"))?;
    std::fs::write(path, bytes)
        .map_err(|err| format!("failed to write tree artifacts {}: {err}", path.display()))
}

pub fn read_tree_artifacts_msgpack<M: LlmModelMarker, S: DatasetSplit>(
    path: impl AsRef<Path>,
) -> Result<Vec<TreeArtifact<M, S>>, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|err| format!("failed to read tree artifacts {}: {err}", path.display()))?;
    rmp_serde::from_slice(&bytes).map_err(|err| {
        format!(
            "failed to deserialize tree artifacts {}: {err}",
            path.display()
        )
    })
}

fn parse_done_marker_chunk_index(file_name: &str) -> Option<usize> {
    let suffix = "_done";
    if !file_name.starts_with("chunk_") || !file_name.ends_with(suffix) {
        return None;
    }
    file_name["chunk_".len()..file_name.len() - suffix.len()]
        .parse::<usize>()
        .ok()
}

fn parse_msgpack_chunk_index(file_name: &str) -> Option<usize> {
    let suffix = ".msgpack";
    if !file_name.starts_with("chunk_") || !file_name.ends_with(suffix) {
        return None;
    }
    file_name["chunk_".len()..file_name.len() - suffix.len()]
        .parse::<usize>()
        .ok()
}

pub fn read_available_tree_artifact_chunks<M: LlmModelMarker, S: DatasetSplit>(
    tree_artifacts_path: impl AsRef<Path>,
) -> Result<Vec<TreeArtifact<M, S>>, String> {
    let path = tree_artifacts_path.as_ref();
    if path.is_file() {
        return read_tree_artifacts_msgpack(path);
    }
    let entries = std::fs::read_dir(path).map_err(|err| {
        format!(
            "failed to read tree artifact directory {}: {}",
            path.display(),
            err
        )
    })?;
    let mut chunk_indices = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read tree artifact directory entry in {}: {}",
                path.display(),
                err
            )
        })?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if let Some(chunk_index) = parse_msgpack_chunk_index(file_name) {
            chunk_indices.push(chunk_index);
        }
    }
    chunk_indices.sort_unstable();
    chunk_indices.dedup();
    let mut artifacts = Vec::new();
    for chunk_index in chunk_indices {
        let chunk_path = tree_artifacts_oneshot_chunk_path(&path.to_string_lossy(), chunk_index);
        let mut chunk_artifacts = read_tree_artifacts_msgpack::<M, S>(&chunk_path)?;
        artifacts.append(&mut chunk_artifacts);
    }
    Ok(artifacts)
}

pub fn read_marked_tree_artifact_chunks<M: LlmModelMarker, S: DatasetSplit>(
    tree_artifacts_path: impl AsRef<Path>,
) -> Result<Vec<TreeArtifact<M, S>>, String> {
    let path = tree_artifacts_path.as_ref();
    if path.is_file() {
        return read_tree_artifacts_msgpack(path);
    }
    let entries = std::fs::read_dir(path).map_err(|err| {
        format!(
            "failed to read tree artifact directory {}: {}",
            path.display(),
            err
        )
    })?;
    let mut chunk_indices = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read tree artifact directory entry in {}: {}",
                path.display(),
                err
            )
        })?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if let Some(chunk_index) = parse_done_marker_chunk_index(file_name) {
            chunk_indices.push(chunk_index);
        }
    }
    chunk_indices.sort_unstable();
    chunk_indices.dedup();
    let mut artifacts = Vec::new();
    for chunk_index in chunk_indices {
        let chunk_path = tree_artifacts_oneshot_chunk_path(&path.to_string_lossy(), chunk_index);
        let mut chunk_artifacts = read_tree_artifacts_msgpack::<M, S>(&chunk_path)?;
        artifacts.append(&mut chunk_artifacts);
    }
    Ok(artifacts)
}

pub fn read_tree_judgments_jsonl(path: impl AsRef<Path>) -> Result<Vec<TreeJudgment>, String> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|err| {
        format!(
            "failed to open tree judgment JSONL {}: {}",
            path.display(),
            err
        )
    })?;
    let reader = std::io::BufRead::lines(std::io::BufReader::new(file));
    let mut judgments = Vec::new();
    for (line_index, line) in reader.enumerate() {
        let line = line.map_err(|err| {
            format!(
                "failed to read tree judgment JSONL {} line {}: {}",
                path.display(),
                line_index + 1,
                err
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let judgment = serde_json::from_str::<TreeJudgment>(&line).map_err(|err| {
            format!(
                "failed to parse tree judgment JSONL {} line {}: {}",
                path.display(),
                line_index + 1,
                err
            )
        })?;
        judgments.push(judgment);
    }
    Ok(judgments)
}

pub fn load_tree_judged_artifacts<M: LlmModelMarker, S: DatasetSplit>(
    tree_artifacts_path: impl AsRef<Path>,
    tree_judgment_jsonl_path: impl AsRef<Path>,
) -> Result<Vec<TreeJudged<M, S>>, String> {
    let artifacts = read_marked_tree_artifact_chunks::<M, S>(tree_artifacts_path.as_ref())?;
    load_tree_judged_artifacts_from_tree_artifacts(artifacts, tree_judgment_jsonl_path)
}

pub fn load_available_tree_judged_artifacts<M: LlmModelMarker, S: DatasetSplit>(
    tree_artifacts_path: impl AsRef<Path>,
    tree_judgment_jsonl_path: impl AsRef<Path>,
) -> Result<Vec<TreeJudged<M, S>>, String> {
    let artifacts = read_available_tree_artifact_chunks::<M, S>(tree_artifacts_path.as_ref())?;
    load_tree_judged_artifacts_from_tree_artifacts(artifacts, tree_judgment_jsonl_path)
}

fn load_tree_judged_artifacts_from_tree_artifacts<M: LlmModelMarker, S: DatasetSplit>(
    artifacts: Vec<TreeArtifact<M, S>>,
    tree_judgment_jsonl_path: impl AsRef<Path>,
) -> Result<Vec<TreeJudged<M, S>>, String> {
    let judgments = read_tree_judgments_jsonl(tree_judgment_jsonl_path.as_ref())?;
    let mut judgments_by_artifact_id: BTreeMap<String, TreeJudgment> = judgments
        .into_iter()
        .map(|judgment| (judgment.artifact_id.clone(), judgment))
        .collect();
    let mut judged = Vec::new();
    for artifact in artifacts {
        let judgment = judgments_by_artifact_id
            .remove(&artifact.artifact_id)
            .ok_or_else(|| {
                format!(
                    "missing tree judgment for artifact {} in {}",
                    artifact.artifact_id,
                    tree_judgment_jsonl_path.as_ref().display()
                )
            })?;
        judged.push(TreeJudged::new(artifact, judgment)?);
    }
    Ok(judged)
}
