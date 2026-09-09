use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Cursor, Read},
    path::PathBuf,
};

use clap::Parser;
use credit_assignment::{
    hybrid_dataset::Training,
    llm_model::{LlmModelMarker, LlmModelName, MyTokenizer, Qwen25_7B},
    rollout_config::TrainingAdvantagePolicy,
    training_set::{
        DirectTrainingTrajectory, reconstruct_all_training_trajectories_from_tree_judged,
    },
    tree::SegmentContent,
    tree_artifact::{TreeArtifact, load_tree_judged_artifacts},
};
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(author, version, about = "Extract trajectory case-study examples")]
struct Args {
    #[arg(value_enum, long, default_value = "qwen25")]
    model_cli_name: LlmModelName,
    #[arg(long)]
    input: Vec<PathBuf>,
    #[arg(long)]
    tree_input: Vec<PathBuf>,
    #[arg(long)]
    tree_judgment_jsonl_path: Vec<PathBuf>,
    #[arg(long)]
    label: Vec<String>,
    #[arg(long)]
    question_flat_id: Option<usize>,
    #[arg(long)]
    leaf_segment_id: Option<usize>,
    #[arg(long, default_value_t = 20)]
    max_candidates: usize,
    #[arg(long, default_value_t = 2200)]
    max_decoded_chars: usize,
    #[arg(long, default_value_t = false)]
    require_python_tool_and_later_positive: bool,
    #[arg(long, default_value_t = false)]
    require_nonzero_advantage: bool,
    #[arg(long, default_value_t = false)]
    list_tree_tool_response_leaves: bool,
    #[arg(long, default_value_t = false)]
    require_tree_tool_response_leaf: bool,
    #[arg(long, default_value_t = false)]
    reconstruct_from_tree_judged: bool,
    #[arg(long, default_value = "tree-mappo-posterior")]
    training_advantage_policy: TrainingAdvantagePolicy,
    #[arg(long, default_value_t = false)]
    positive_advantage_only: bool,
}

#[derive(Clone)]
struct AdvantageRun {
    start: usize,
    end: usize,
    advantage: f32,
}

fn read_training_trajectories<M: LlmModelMarker>(
    input_path: &PathBuf,
) -> Vec<DirectTrainingTrajectory<M>> {
    let mut bytes = Vec::new();
    File::open(input_path)
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", input_path.display()))
        .read_to_end(&mut bytes)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", input_path.display()));
    let mut cursor = Cursor::new(bytes.as_slice());
    let total_len = bytes.len() as u64;
    let mut trajectories = Vec::new();
    while cursor.position() < total_len {
        let mut deserializer = rmp_serde::Deserializer::new(&mut cursor);
        let trajectory = DirectTrainingTrajectory::<M>::deserialize(&mut deserializer)
            .unwrap_or_else(|err| panic!("failed to deserialize {}: {err}", input_path.display()));
        trajectories.push(trajectory);
    }
    trajectories
}

fn read_tree_artifacts<M: LlmModelMarker>(input_path: &PathBuf) -> Vec<TreeArtifact<M, Training>> {
    let mut bytes = Vec::new();
    File::open(input_path)
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", input_path.display()))
        .read_to_end(&mut bytes)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", input_path.display()));
    rmp_serde::from_slice(&bytes)
        .unwrap_or_else(|err| panic!("failed to deserialize {}: {err}", input_path.display()))
}

fn supervised_advantage_runs<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
) -> Vec<AdvantageRun> {
    let mut runs = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_advantage = 0.0_f32;
    for index in 0..trajectory.input_ids.len() {
        if trajectory.labels[index] == -100 {
            if let Some(start) = current_start.take() {
                runs.push(AdvantageRun {
                    start,
                    end: index,
                    advantage: current_advantage,
                });
            }
            continue;
        }
        let advantage = trajectory.advantages[index];
        match current_start {
            Some(_) if (advantage - current_advantage).abs() < 1.0e-5 => {}
            Some(start) => {
                runs.push(AdvantageRun {
                    start,
                    end: index,
                    advantage: current_advantage,
                });
                current_start = Some(index);
                current_advantage = advantage;
            }
            None => {
                current_start = Some(index);
                current_advantage = advantage;
            }
        }
    }
    if let Some(start) = current_start {
        runs.push(AdvantageRun {
            start,
            end: trajectory.input_ids.len(),
            advantage: current_advantage,
        });
    }
    runs
}

fn has_negative_to_positive_twist(runs: &[AdvantageRun]) -> bool {
    let mut saw_negative = false;
    for run in runs {
        if run.advantage < -0.2 {
            saw_negative = true;
        }
        if saw_negative && run.advantage > 0.2 {
            return true;
        }
    }
    false
}

fn token_index_after_text_marker<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
    marker: &str,
) -> Option<usize> {
    let mut decoded_prefix = String::new();
    for (index, token_id) in trajectory.input_ids.iter().enumerate() {
        decoded_prefix.push_str(&<M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&[
            *token_id,
        ]));
        if decoded_prefix.contains(marker) {
            return Some(index + 1);
        }
    }
    None
}

fn has_python_tool_and_later_positive<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
    runs: &[AdvantageRun],
) -> bool {
    let decoded = <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&trajectory.input_ids);
    if !decoded.contains("```python") {
        return false;
    }
    let Some(output_marker_end) = token_index_after_text_marker(trajectory, "```output") else {
        return false;
    };
    let saw_negative_before_output = runs
        .iter()
        .any(|run| run.start < output_marker_end && run.advantage < -0.1);
    let saw_positive_after_output = runs
        .iter()
        .any(|run| run.start >= output_marker_end && run.advantage > 0.2);
    saw_negative_before_output && saw_positive_after_output
}

fn rgb_for_advantage(advantage: f32, scale: f32) -> (u8, u8, u8) {
    let neutral = (145.0_f32, 125.0_f32, 45.0_f32);
    let safe_scale = scale.max(1.0e-6);
    if advantage < 0.0 {
        let t = (-advantage / safe_scale).clamp(0.0, 1.0);
        let saturated = (130.0_f32, 20.0_f32, 25.0_f32);
        (
            (neutral.0 * (1.0 - t) + saturated.0 * t).round() as u8,
            (neutral.1 * (1.0 - t) + saturated.1 * t).round() as u8,
            (neutral.2 * (1.0 - t) + saturated.2 * t).round() as u8,
        )
    } else {
        let t = (advantage / safe_scale).clamp(0.0, 1.0);
        let saturated = (20.0_f32, 110.0_f32, 55.0_f32);
        (
            (neutral.0 * (1.0 - t) + saturated.0 * t).round() as u8,
            (neutral.1 * (1.0 - t) + saturated.1 * t).round() as u8,
            (neutral.2 * (1.0 - t) + saturated.2 * t).round() as u8,
        )
    }
}

fn latex_escape(text: &str) -> String {
    let mut output = String::new();
    for character in text.chars() {
        match character {
            '\\' => output.push_str("\\textbackslash{}"),
            '{' => output.push_str("\\{"),
            '}' => output.push_str("\\}"),
            '$' => output.push_str("\\$"),
            '&' => output.push_str("\\&"),
            '#' => output.push_str("\\#"),
            '_' => output.push_str("\\_"),
            '%' => output.push_str("\\%"),
            '^' => output.push_str("\\textasciicircum{}"),
            '~' => output.push_str("\\textasciitilde{}"),
            '\n' => output.push_str("\\\\\n"),
            ' ' => output.push('~'),
            '\t' => output.push_str("\\quad{}"),
            _ => output.push(character),
        }
    }
    output
}

fn latex_colored_trajectory<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
    max_chars: usize,
) -> String {
    let mut output = String::from("\\begingroup\\ttfamily\\scriptsize\n");
    let mut current_rgb: Option<(u8, u8, u8)> = None;
    let mut current_text = String::new();
    let mut emitted_chars = 0usize;
    let advantage_scale = trajectory
        .advantages
        .iter()
        .zip(trajectory.labels.iter())
        .filter(|(_, label)| **label != -100)
        .map(|(advantage, _)| advantage.abs())
        .fold(0.0_f32, f32::max);

    for index in 0..trajectory.input_ids.len() {
        if emitted_chars >= max_chars {
            break;
        }
        let token_text =
            <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&[trajectory.input_ids[index]]);
        emitted_chars += token_text.chars().count();
        let token_rgb = if trajectory.labels[index] == -100 {
            None
        } else {
            Some(rgb_for_advantage(
                trajectory.advantages[index],
                advantage_scale,
            ))
        };
        if token_rgb != current_rgb {
            flush_latex_group(&mut output, current_rgb, &current_text);
            current_text.clear();
            current_rgb = token_rgb;
        }
        current_text.push_str(&token_text);
    }
    flush_latex_group(&mut output, current_rgb, &current_text);
    if emitted_chars >= max_chars {
        output.push_str("\\\\\n\\emph{[truncated]}");
    }
    output.push_str("\n\\endgroup");
    output
}

fn flush_latex_group(output: &mut String, rgb: Option<(u8, u8, u8)>, text: &str) {
    if text.is_empty() {
        return;
    }
    let escaped = latex_escape(text);
    if let Some((red, green, blue)) = rgb {
        output.push_str(&format!(
            "\\textcolor[RGB]{{{red},{green},{blue}}}{{{escaped}}}"
        ));
    } else {
        output.push_str(&escaped);
    }
}

fn plain_decoded<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
    max_chars: usize,
) -> String {
    let decoded = <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&trajectory.input_ids);
    decoded.chars().take(max_chars).collect()
}

fn decoded_prompt_prefix<M: LlmModelMarker>(trajectory: &DirectTrainingTrajectory<M>) -> String {
    let prompt_end = trajectory
        .labels
        .iter()
        .position(|label| *label != -100)
        .unwrap_or(trajectory.labels.len());
    <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&trajectory.input_ids[..prompt_end])
}

fn decoded_supervised_model_text<M: LlmModelMarker>(
    trajectory: &DirectTrainingTrajectory<M>,
) -> String {
    let mut output = String::new();
    for index in 0..trajectory.input_ids.len() {
        if trajectory.labels[index] != -100 {
            output.push_str(&<M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&[
                trajectory.input_ids[index],
            ]));
        }
    }
    output
}

fn latex_verbatim_block(text: &str, max_chars: usize) -> String {
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        truncated.push_str("\n[truncated]");
    }
    format!(
        "\\begingroup\\ttfamily\\scriptsize\n{}\\endgroup",
        latex_escape(&truncated)
    )
}

fn print_trajectory<M: LlmModelMarker>(
    label: &str,
    index: usize,
    trajectory: &DirectTrainingTrajectory<M>,
    max_decoded_chars: usize,
) {
    let runs = supervised_advantage_runs(trajectory);
    println!(
        "BEGIN_TRAJECTORY label={} index={} flat_id={} dataset={} question_id={} leaf_segment={:?} len={} avg_abs_adv={:.6}",
        label,
        index,
        trajectory.question.flat_id.0,
        trajectory.question.dataset_name,
        trajectory.question.question_id,
        trajectory.leaf_segment_id,
        trajectory.input_ids.len(),
        trajectory.average_absolute_segment_advantage,
    );
    println!("QUESTION:\n{}", trajectory.question.question);
    println!(
        "PROMPT_PREFIX_RAW:\n{}",
        decoded_prompt_prefix(trajectory)
            .chars()
            .take(max_decoded_chars)
            .collect::<String>()
    );
    println!(
        "MODEL_SUPERVISED_RAW:\n{}",
        decoded_supervised_model_text(trajectory)
            .chars()
            .take(max_decoded_chars)
            .collect::<String>()
    );
    println!(
        "LATEX_PROMPT_PREFIX_RAW:\n{}",
        latex_verbatim_block(&decoded_prompt_prefix(trajectory), max_decoded_chars)
    );
    println!(
        "LATEX_MODEL_SUPERVISED_RAW:\n{}",
        latex_verbatim_block(
            &decoded_supervised_model_text(trajectory),
            max_decoded_chars
        )
    );
    println!("ADVANTAGE_RUNS:");
    for run in &runs {
        let decoded = <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(
            &trajectory.input_ids[run.start..run.end],
        );
        let clipped = decoded.replace('\n', "\\n");
        println!(
            "  tokens=[{}, {}) advantage={:.4} text={}",
            run.start,
            run.end,
            run.advantage,
            clipped.chars().take(240).collect::<String>()
        );
    }
    println!(
        "LATEX_COLORED:\n{}",
        latex_colored_trajectory(trajectory, max_decoded_chars)
    );
    println!(
        "PLAIN_DECODED:\n{}",
        plain_decoded(trajectory, max_decoded_chars)
    );
    println!("END_TRAJECTORY");
}

fn print_tree_path<M: LlmModelMarker>(label: &str, artifact: &TreeArtifact<M, Training>) {
    println!(
        "BEGIN_TREE_PATH label={} artifact_id={} flat_id={} dataset={} question_id={} use_tool={} root={:?}",
        label,
        artifact.artifact_id,
        artifact.question.flat_id.0,
        artifact.question.dataset_name,
        artifact.question.question_id,
        artifact.use_tool,
        artifact.root_segment_id,
    );
    let Some(leaf_segment_id) = artifact
        .leaf_answers
        .first()
        .map(|leaf| leaf.leaf_segment_id)
    else {
        println!("NO_LEAF_ANSWERS");
        println!("END_TREE_PATH");
        return;
    };
    println!("FIRST_LEAF_SEGMENT={:?}", leaf_segment_id);
    for segment in &artifact.segments {
        for (content_index, content) in segment.content.iter().enumerate() {
            let kind = match content {
                SegmentContent::Prompt(_) => "prompt",
                SegmentContent::ReasoningOrToolCall { complete, .. } => {
                    if *complete {
                        "reasoning_or_tool_call_complete"
                    } else {
                        "reasoning_or_tool_call_incomplete"
                    }
                }
                SegmentContent::ToolResponse(_) => "tool_response",
            };
            let decoded = <M::Tokenizer as MyTokenizer<M>>::decode_i32_ids(&content.tokens());
            println!(
                "TREE_CONTENT segment={:?} parent={:?} content={} kind={} tokens={} text={:?}",
                segment.segment_id,
                segment.parent_id,
                content_index,
                kind,
                content.tokens().len(),
                decoded.chars().take(700).collect::<String>(),
            );
        }
    }
    println!("END_TREE_PATH");
}

fn tree_path_segment_ids<M: LlmModelMarker>(
    artifact: &TreeArtifact<M, Training>,
    leaf_segment_id: credit_assignment::tree::SegmentId,
) -> Vec<credit_assignment::tree::SegmentId> {
    let segments_by_id = artifact
        .segments
        .iter()
        .map(|segment| (segment.segment_id, segment))
        .collect::<BTreeMap<_, _>>();
    let mut path = Vec::new();
    let mut current = Some(leaf_segment_id);
    while let Some(segment_id) = current {
        path.push(segment_id);
        current = segments_by_id
            .get(&segment_id)
            .and_then(|segment| segment.parent_id);
    }
    path.reverse();
    path
}

fn tree_path_has_tool_response<M: LlmModelMarker>(
    artifact: &TreeArtifact<M, Training>,
    leaf_segment_id: credit_assignment::tree::SegmentId,
) -> bool {
    let segments_by_id = artifact
        .segments
        .iter()
        .map(|segment| (segment.segment_id, segment))
        .collect::<BTreeMap<_, _>>();
    tree_path_segment_ids(artifact, leaf_segment_id)
        .iter()
        .filter_map(|segment_id| segments_by_id.get(segment_id))
        .any(|segment| {
            segment
                .content
                .iter()
                .any(|content| matches!(content, SegmentContent::ToolResponse(_)))
        })
}

fn print_tree_tool_response_leaves<M: LlmModelMarker>(
    label: &str,
    artifact: &TreeArtifact<M, Training>,
) -> bool {
    let matching_leaves = artifact
        .leaf_answers
        .iter()
        .filter(|leaf| tree_path_has_tool_response(artifact, leaf.leaf_segment_id))
        .map(|leaf| format!("{:?}", leaf.leaf_segment_id))
        .collect::<Vec<_>>();
    if matching_leaves.is_empty() {
        return false;
    }
    println!(
        "TREE_TOOL_RESPONSE_LEAVES label={} flat_id={} dataset={} question_id={} leaves={}",
        label,
        artifact.question.flat_id.0,
        artifact.question.dataset_name,
        artifact.question.question_id,
        matching_leaves.join(","),
    );
    println!("QUESTION:\n{}", artifact.question.question);
    true
}

fn run_qwen25(args: Args) {
    if args.reconstruct_from_tree_judged {
        assert_eq!(
            args.tree_input.len(),
            args.tree_judgment_jsonl_path.len(),
            "--reconstruct-from-tree-judged requires one --tree-judgment-jsonl-path per --tree-input"
        );
    }
    let labels = if args.label.is_empty() {
        args.input
            .iter()
            .chain(args.tree_input.iter())
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
    } else {
        args.label.clone()
    };
    assert!(
        labels.len() == args.input.len()
            || labels.len() == args.input.len() + args.tree_input.len(),
        "--label count must equal --input count, or --input plus --tree-input count"
    );

    let mut tree_tool_response_keys = BTreeSet::new();
    if args.require_tree_tool_response_leaf {
        for tree_input_path in &args.tree_input {
            let artifacts = read_tree_artifacts::<Qwen25_7B>(tree_input_path);
            for artifact in artifacts.iter() {
                for leaf in artifact.leaf_answers.iter() {
                    if tree_path_has_tool_response(artifact, leaf.leaf_segment_id) {
                        tree_tool_response_keys
                            .insert((artifact.question.flat_id.0, leaf.leaf_segment_id.0));
                    }
                }
            }
        }
        eprintln!(
            "loaded {} tree tool-response trajectory keys",
            tree_tool_response_keys.len()
        );
    }

    let mut candidate_question_ids = BTreeSet::new();
    for (input_index, input_path) in args.input.iter().enumerate() {
        let trajectories = read_training_trajectories::<Qwen25_7B>(input_path);
        let mut printed = 0usize;
        for (trajectory_index, trajectory) in trajectories.iter().enumerate() {
            if let Some(question_flat_id) = args.question_flat_id {
                if trajectory.question.flat_id.0 != question_flat_id {
                    continue;
                }
            }
            if let Some(leaf_segment_id) = args.leaf_segment_id {
                if trajectory.leaf_segment_id.0 != leaf_segment_id {
                    continue;
                }
            }
            if args.require_tree_tool_response_leaf
                && !tree_tool_response_keys
                    .contains(&(trajectory.question.flat_id.0, trajectory.leaf_segment_id.0))
            {
                continue;
            }
            let runs = supervised_advantage_runs(trajectory);
            let matches_search = if args.require_python_tool_and_later_positive {
                has_python_tool_and_later_positive(trajectory, &runs)
            } else if args.require_nonzero_advantage {
                runs.iter().any(|run| run.advantage.abs() > 1.0e-4)
            } else {
                has_negative_to_positive_twist(&runs)
            };
            if args.question_flat_id.is_some() || matches_search {
                candidate_question_ids.insert(trajectory.question.flat_id.0);
                print_trajectory(
                    &labels[input_index],
                    trajectory_index,
                    trajectory,
                    args.max_decoded_chars,
                );
                printed += 1;
                if printed >= args.max_candidates {
                    break;
                }
            }
        }
    }
    for (tree_input_index, tree_input_path) in args.tree_input.iter().enumerate() {
        let label_index = args.input.len() + tree_input_index;
        let label = labels
            .get(label_index)
            .or_else(|| labels.get(tree_input_index))
            .map(String::as_str)
            .unwrap_or("tree-artifact");
        if args.reconstruct_from_tree_judged {
            let tree_judged_artifacts = load_tree_judged_artifacts::<Qwen25_7B, Training>(
                tree_input_path,
                &args.tree_judgment_jsonl_path[tree_input_index],
            )
            .unwrap_or_else(|err| {
                panic!(
                    "failed to load judged tree artifacts from {} and {}: {err}",
                    tree_input_path.display(),
                    args.tree_judgment_jsonl_path[tree_input_index].display()
                )
            });
            let mut printed = 0usize;
            for tree_judged in tree_judged_artifacts.iter() {
                if let Some(question_flat_id) = args.question_flat_id {
                    if tree_judged.tree.question.flat_id.0 != question_flat_id {
                        continue;
                    }
                }
                let reconstructed = reconstruct_all_training_trajectories_from_tree_judged(
                    tree_judged,
                    args.training_advantage_policy,
                    args.positive_advantage_only,
                );
                for (trajectory_index, trajectory) in reconstructed {
                    if let Some(leaf_segment_id) = args.leaf_segment_id {
                        if trajectory.leaf_segment_id.0 != leaf_segment_id {
                            continue;
                        }
                    }
                    if args.require_tree_tool_response_leaf
                        && !tree_path_has_tool_response(
                            &tree_judged.tree,
                            trajectory.leaf_segment_id,
                        )
                    {
                        continue;
                    }
                    let runs = supervised_advantage_runs(&trajectory);
                    let matches_search = if args.require_python_tool_and_later_positive {
                        has_python_tool_and_later_positive(&trajectory, &runs)
                    } else if args.require_nonzero_advantage {
                        runs.iter().any(|run| run.advantage.abs() > 1.0e-4)
                    } else {
                        has_negative_to_positive_twist(&runs)
                    };
                    if args.question_flat_id.is_some() || matches_search {
                        candidate_question_ids.insert(trajectory.question.flat_id.0);
                        print_trajectory(
                            label,
                            trajectory_index,
                            &trajectory,
                            args.max_decoded_chars,
                        );
                        printed += 1;
                        if printed >= args.max_candidates {
                            break;
                        }
                    }
                }
                if printed >= args.max_candidates {
                    break;
                }
            }
            continue;
        }

        let artifacts = read_tree_artifacts::<Qwen25_7B>(tree_input_path);
        let mut printed = 0usize;
        for artifact in artifacts.iter() {
            if let Some(question_flat_id) = args.question_flat_id {
                if artifact.question.flat_id.0 != question_flat_id {
                    continue;
                }
            }
            if let Some(leaf_segment_id) = args.leaf_segment_id {
                if !artifact
                    .leaf_answers
                    .iter()
                    .any(|leaf| leaf.leaf_segment_id.0 == leaf_segment_id)
                {
                    continue;
                }
            }
            let did_print = if args.list_tree_tool_response_leaves {
                print_tree_tool_response_leaves(label, artifact)
            } else {
                print_tree_path(label, artifact);
                true
            };
            if did_print {
                printed += 1;
            }
            if printed >= args.max_candidates {
                break;
            }
        }
    }
    println!("CANDIDATE_QUESTION_IDS {:?}", candidate_question_ids);
}

fn main() {
    let args = Args::parse();
    match args.model_cli_name {
        LlmModelName::Qwen25_7b => run_qwen25(args),
        other => panic!(
            "only qwen25 is implemented for now, got {}",
            other.cli_name()
        ),
    }
}
