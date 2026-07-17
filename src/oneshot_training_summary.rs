use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use crate::oneshot_utils::oneshot_aggregated_summary_path;
use research_utility::progress_text_logger::{log_info, log_warning};
use serde::Deserialize;

pub fn ensure_parent_dir_exists(file_path: &str) -> Result<(), String> {
    let Some(parent) = Path::new(file_path).parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(parent).map_err(|err| {
        format!(
            "Failed to create parent directory {}: {}",
            parent.display(),
            err
        )
    })
}

pub fn derive_phase_log_path(base_path: &str, phase_suffix: &str) -> String {
    let path = Path::new(base_path);
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("log.txt");
    let derived_file_name = if let Some((stem, ext)) = file_name.rsplit_once('.') {
        format!("{stem}_{phase_suffix}.{ext}")
    } else {
        format!("{file_name}_{phase_suffix}")
    };
    path.with_file_name(derived_file_name)
        .to_string_lossy()
        .into_owned()
}

pub fn write_training_summary(
    summary_parent_dir: &str,
    latest_epoch: usize,
    num_oneshot_epochs: usize,
    validation_accuracies: &BTreeMap<usize, (f32, f32, f32, f32)>,
    training_throughputs: &BTreeMap<usize, f32>,
    training_samples_trained: &BTreeMap<usize, usize>,
    training_longest_non_oom_trajectory_lengths: &BTreeMap<usize, usize>,
) {
    std::fs::create_dir_all(summary_parent_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create training summary parent dir {}: {}",
            summary_parent_dir, err
        )
    });

    let mut accuracies_json: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (epoch, (avg, deepmath, math, _numinamath)) in validation_accuracies {
        accuracies_json.insert(
            format!("epoch_{}", epoch),
            serde_json::json!({
                "avg": avg,
                "deepmath": deepmath,
                "math": math,
            }),
        );
    }

    let mut throughputs_json: BTreeMap<String, f32> = BTreeMap::new();
    for (epoch, throughput) in training_throughputs {
        throughputs_json.insert(format!("epoch_{}", epoch), *throughput);
    }
    let mut samples_json: BTreeMap<String, usize> = BTreeMap::new();
    for (epoch, samples_trained) in training_samples_trained {
        samples_json.insert(format!("epoch_{}", epoch), *samples_trained);
    }
    let mut lengths_json: BTreeMap<String, usize> = BTreeMap::new();
    for (epoch, longest_length) in training_longest_non_oom_trajectory_lengths {
        lengths_json.insert(format!("epoch_{}", epoch), *longest_length);
    }

    let latest_epoch_data = {
        let (avg, deepmath, math, _numinamath) = validation_accuracies
            .get(&latest_epoch)
            .map(|(a, b, c, d)| (*a, *b, *c, *d))
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        let training_epoch = latest_epoch.checked_sub(1);
        let throughput = training_epoch
            .and_then(|epoch| training_throughputs.get(&epoch).copied())
            .unwrap_or(0.0);
        let samples_trained = training_epoch
            .and_then(|epoch| training_samples_trained.get(&epoch).copied())
            .unwrap_or(0);
        let longest_non_oom_trajectory_length = training_epoch
            .and_then(|epoch| {
                training_longest_non_oom_trajectory_lengths
                    .get(&epoch)
                    .copied()
            })
            .unwrap_or(0);
        serde_json::json!({
            "epoch": latest_epoch,
            "validation_accuracy": {
                "avg": avg,
                "deepmath": deepmath,
                "math": math,
            },
            "training_throughput": throughput,
            "training_samples_trained": samples_trained,
            "longest_non_oom_trajectory_length": longest_non_oom_trajectory_length,
        })
    };

    let epoch_output_path = Path::new(summary_parent_dir).join(format!(
        "oneshot_per_epoch_summary_epoch_{}.json",
        latest_epoch
    ));
    std::fs::write(
        &epoch_output_path,
        serde_json::to_string_pretty(&latest_epoch_data).unwrap() + "\n",
    )
    .unwrap_or_else(|err| {
        panic!(
            "Failed to write per-epoch oneshot summary to {}: {}",
            epoch_output_path.display(),
            err
        )
    });
    log_info(format!(
        "Wrote per-epoch oneshot summary to {}",
        epoch_output_path.display()
    ));

    let accumulated_path = oneshot_aggregated_summary_path(summary_parent_dir);
    let payload = serde_json::json!({
        "latest_epoch": latest_epoch,
        "num_oneshot_epochs": num_oneshot_epochs,
        "validation_accuracies": accuracies_json,
        "training_throughputs": throughputs_json,
        "training_samples_trained": samples_json,
        "training_longest_non_oom_trajectory_lengths": lengths_json,
    });
    std::fs::write(
        &accumulated_path,
        serde_json::to_string_pretty(&payload).unwrap() + "\n",
    )
    .unwrap_or_else(|err| {
        panic!(
            "Failed to write aggregated oneshot summary to {}: {}",
            accumulated_path.display(),
            err
        )
    });
    log_info(format!(
        "Wrote aggregated oneshot summary to {}",
        accumulated_path.display()
    ));
}

#[derive(Deserialize)]
struct PythonTrainingSummary {
    samples_trained: usize,
    #[serde(default)]
    samples_trained_this_run: usize,
    total_training_time_sec: f32,
    #[serde(default)]
    longest_non_oom_trajectory_length: usize,
}

pub struct OneshotTrainingEpochStats {
    pub throughputs: BTreeMap<usize, f32>,
    pub samples_trained: BTreeMap<usize, usize>,
    pub longest_non_oom_trajectory_lengths: BTreeMap<usize, usize>,
}

pub fn read_oneshot_training_epoch_stats(
    oneshot_model_output_root: &str,
    num_oneshot_epochs: usize,
) -> OneshotTrainingEpochStats {
    let mut throughputs: BTreeMap<usize, f32> = BTreeMap::new();
    let mut samples_trained_by_epoch: BTreeMap<usize, usize> = BTreeMap::new();
    let mut longest_lengths_by_epoch: BTreeMap<usize, usize> = BTreeMap::new();
    let mut previous_samples_trained = 0usize;
    for epoch in 0..num_oneshot_epochs {
        let summary_path = Path::new(oneshot_model_output_root)
            .join(format!("oneshot_epoch_{}/training_summary.json", epoch + 1));
        let summary = fs::read_to_string(&summary_path)
            .ok()
            .and_then(|content| serde_json::from_str::<PythonTrainingSummary>(&content).ok());
        let (samples_this_run, throughput, longest_non_oom_trajectory_length) =
            if let Some(summary) = summary {
                let samples_this_run = if summary.samples_trained_this_run > 0 {
                    summary.samples_trained_this_run
                } else {
                    summary
                        .samples_trained
                        .saturating_sub(previous_samples_trained)
                };
                previous_samples_trained = summary.samples_trained;
                let throughput = if summary.total_training_time_sec <= f32::EPSILON {
                    0.0
                } else {
                    samples_this_run as f32 / summary.total_training_time_sec
                };
                (
                    samples_this_run,
                    throughput,
                    summary.longest_non_oom_trajectory_length,
                )
            } else {
                (0usize, 0.0f32, 0usize)
            };
        throughputs.insert(epoch, throughput);
        samples_trained_by_epoch.insert(epoch, samples_this_run);
        longest_lengths_by_epoch.insert(epoch, longest_non_oom_trajectory_length);
    }
    OneshotTrainingEpochStats {
        throughputs,
        samples_trained: samples_trained_by_epoch,
        longest_non_oom_trajectory_lengths: longest_lengths_by_epoch,
    }
}

pub fn read_existing_validation_summary(
    summary_parent_dir: &str,
) -> (
    HashSet<usize>,
    BTreeMap<usize, (f32, f32, f32, f32)>,
) {
    let aggregated_path = oneshot_aggregated_summary_path(summary_parent_dir);
    if !aggregated_path.exists() {
        return (HashSet::new(), BTreeMap::new());
    }

    let mut already_validated_epochs = HashSet::new();
    let mut validation_accuracies = BTreeMap::new();
    let Ok(content) = std::fs::read_to_string(&aggregated_path) else {
        log_warning(format!(
            "Failed to read aggregated summary at {}; validation resume metadata will be ignored",
            aggregated_path.display()
        ));
        return (already_validated_epochs, validation_accuracies);
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) else {
        log_warning(format!(
            "Failed to parse aggregated summary at {}; validation resume metadata will be ignored",
            aggregated_path.display()
        ));
        return (already_validated_epochs, validation_accuracies);
    };

    if let Some(acc_map) = parsed.get("validation_accuracies").and_then(|v| v.as_object()) {
        for (key, value) in acc_map {
            let Some(epoch_str) = key.strip_prefix("epoch_") else {
                continue;
            };
            let Ok(epoch) = epoch_str.parse::<usize>() else {
                continue;
            };
            already_validated_epochs.insert(epoch);
            let avg = value.get("avg").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let deepmath = value.get("deepmath").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let math = value.get("math").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            validation_accuracies.insert(epoch, (avg, deepmath, math, 0.0));
        }
    }

    (already_validated_epochs, validation_accuracies)
}

pub fn detect_trained_oneshot_epochs(
    oneshot_model_output_root: &str,
    num_oneshot_epochs: usize,
) -> Vec<usize> {
    let mut trained_epochs = Vec::new();
    for epoch in 1..=num_oneshot_epochs {
        let model_dir = Path::new(oneshot_model_output_root).join(format!("oneshot_epoch_{epoch}/model"));
        if model_dir.exists() {
            trained_epochs.push(epoch);
        }
    }
    trained_epochs
}

pub fn prune_non_best_oneshot_models(
    oneshot_model_output_root: &str,
    candidate_epochs: &[usize],
    validation_accuracies: &BTreeMap<usize, (f32, f32, f32, f32)>,
) {
    let Some((best_epoch, best_accuracy)) = validation_accuracies
        .iter()
        .filter(|(epoch, _)| **epoch > 0)
        .max_by(|left, right| left.1 .0.total_cmp(&right.1 .0))
        .map(|(epoch, accuracies)| (*epoch, accuracies.0))
    else {
        log_info(
            "No validated trained epoch is available; skipping post-validation model pruning",
        );
        return;
    };

    log_info(format!(
        "Post-validation model pruning will keep oneshot_epoch_{} (best avg accuracy {:.6}) and remove other trained epoch snapshots",
        best_epoch, best_accuracy
    ));

    for &epoch in candidate_epochs {
        if epoch == best_epoch {
            continue;
        }
        let epoch_dir = Path::new(oneshot_model_output_root).join(format!("oneshot_epoch_{epoch}"));
        if !epoch_dir.exists() {
            continue;
        }
        match std::fs::remove_dir_all(&epoch_dir) {
            Ok(()) => log_info(format!(
                "Pruned non-best oneshot model snapshot at {}",
                epoch_dir.display()
            )),
            Err(err) => log_warning(format!(
                "Failed to prune non-best oneshot model snapshot at {}: {}",
                epoch_dir.display(),
                err
            )),
        }
    }
}
