# Modal experiments
To run the experiments on Modal, go to `scripts/orchestrator_modal` folder and select one of the scripts. For example: `scripts/orchestrator_modal/qwen25_std.sh`.

Then we do `bash scripts/orchestrator_modal/qwen25_std.sh` to deploy the experiment on Modal. You can run multiple deployment scripts at the same time, but they will mostly run in serial since the deployment involves writing a json config file, uploading the repository containing the json file to Modal, and then deleting the local json file. The second deployment script will not run until the first one has completed. It is normal to see the execution pauses at `Waiting for config file lock...` if there are other deployment scripts already running.

# Modal progress inspection
The first thing to check after deployment is https://modal.com/apps/glad-lab/main. Click into the app you just deployed, and it will show the logs.

# Debugging
Go to https://modal.com/storage/glad-lab/main to see the volumes.

The corresponding volume should have a name like `credit-assignment-qwen2-5-7b-std`.

There should be `small_files`, `medium_files`, and `large_files` directories.

`small_files` contains the inference wrapper log, training wrapper log, orchestration progress and tui log. The last one requires `bash tui.sh` to run.

`medium_files` contains the inference and training action logs (the trees). You can use `cargo run --bin bin_browse_trees` to browse them.

`large_files` contains the model weights and checkpoints.

## Downloading volume folders to local machine
We have:
- `scripts/download_modal_small_files.py`
- `scripts/download_modal_medium_files.py`
- `scripts/download_modal_large_files.py`

For example to download the small files folder, run:

```bash
uv run scripts/download_modal_small_files.py --model-cli-name qwen2.5-7b --config-nickname std
```

The `--model-cli-name` options can be found in `src/llm_model/llm_model_name.rs`.

`--config-nickname` is the one in `scripts/orchestrator_modal` folder scripts.

## Replay the training progress
1. Locate the downloaded small files folder. In it, there is a `tui_log.bin` file. For example: `modal_downloads/qwen2.5-7b/std/small_files/qwen2.5-7b/std/tui_log.bin`.
2. Run `bash tui.sh [tui_log.bin path]`

## Browse the action logs that form the trees
1. Locate the action log files in medium files folder. For example: // to do
2. Run `cargo run --bin bin_browse_trees -- --action-logs-path [action log files path]`

