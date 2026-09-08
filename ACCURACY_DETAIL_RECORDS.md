# Accuracy Detail Records

This document defines the required artifact contract for validation and testing accuracy records.

## Policy

- Every held-out validation score must preserve per-trial, per-dataset accuracies in addition to aggregate averages.
- Every serious testing score must preserve per-trial, per-dataset accuracies in addition to aggregate averages.
- Accuracy details must be written to an experiment-specific path and to a backup path under `accuracy_details_backup/`.
- Epochs other than `0` or multiples of `10` are deprecated for paper tables unless explicitly marked as legacy exploratory results.
- Final paper results should use full dataset coverage and enough trials to match the current experiment policy.

## Validation Artifacts

For validation epoch `E`, the score phase writes:

- Canonical detail file:
  `/work/hdd/bhph/zluo8/credit_assignment/results/small_files/<model>/<config>/epoch_E/validation_accuracy_details.json`
- Backup detail file:
  `/work/hdd/bhph/zluo8/credit_assignment/results/small_files/<model>/<config>/accuracy_details_backup/validation_epoch_E.json`

Each file contains:

- `model_cli_name`, `config_nickname`, `epoch`, and `dataset_split`.
- `requested_num_rollout_trials` and `scored_num_rollout_trials`.
- `per_trial`, with one record per successfully scored rollout trial.
- Per trial: average accuracy, DeepMath accuracy, MATH accuracy, NuminaMath accuracy, judged tree count, judged trajectory count, weighted wins, and weighted total plays.
- `aggregate`, matching the existing aggregate validation score fields.

## Testing Artifacts

For testing epoch `E`, the score phase writes:

- Canonical score file:
  `/work/hdd/bhph/zluo8/credit_assignment/results/small_files/<model>/<config>/test_accuracy_epoch_E.json`
- Backup score file:
  `/work/hdd/bhph/zluo8/credit_assignment/results/small_files/<model>/<config>/accuracy_details_backup/test_accuracy_epoch_E.json`

The testing score schema already stores the needed detail:

- `per_dataset.<dataset>.accuracy_values`: individual per-trial accuracy values for that dataset.
- `per_dataset.<dataset>.mean_accuracy`: mean across trials for that dataset.
- `per_dataset.<dataset>.confidence_interval_half_width`: confidence interval over trial values.
- `macro_average.accuracy_values`: equal-dataset macro average for each trial index when available.
- `macro_average.mean_accuracy`: average of macro trial values.

## Existing Results

- Existing testing JSON files already contain per-dataset trial arrays; the new backup path is created by future score jobs.
- Existing validation aggregate files do not always preserve per-trial details. For final paper use, rerun the validation score phase with the current code so `validation_accuracy_details.json` and its backup are produced.
- When auditing prior results, treat any score missing `accuracy_details_backup/` as recoverable only if the corresponding tree artifacts and judgment JSONL files are still present.
