# Agent Instructions

## Notification on job completion
After every task or job that you complete, call the Pushover notification script with a concise one-line summary of what was done:

```sh
python research-utility/scripts/pushover_notify.py "<one-line summary>"
```

Keep the message brief and descriptive — e.g. `"Fixed shape mismatch in attention layer"` or `"Ran full evaluation suite, all 42 tests pass"`.

## Delta SSH login workflow
When working with the `delta` host, prefer a persistent SSH session tool such as Zed's `ssh-tmux` MCP server instead of the stateless terminal.

Working login sequence:
1. Open an SSH session to host alias `delta`.
2. Wait for the password prompt.
3. Tell the user to attach locally with `tmux attach -t mcp-ssh`.
4. Let the user type the password directly in the attached tmux session.
5. At the Duo menu, send `1` for `Duo Push` or let the user type it in the attached tmux session.
6. Wait for the user to approve the push notification.
7. Confirm the remote shell prompt appears before running commands.
8. Tell the user to detach from tmux with `Ctrl+B`, then `D`.

Notes:
- Do not claim Delta access is unattended; a human must still complete MFA.
- Keep the session open while waiting for Duo approval.
- Use `tmux attach -t mcp-ssh` when the user wants to type secrets without sending them in chat.
- Tell the user to detach with `Ctrl+B`, then `D` after finishing interactive input.
- Never store passwords or MFA codes in repo files.

## Delta job monitoring
When monitoring long-running SLURM jobs on Delta via `ssh-tmux`:

- **Do NOT use `sleep` inside `send_command`.** The tmux session handles long-running commands poorly — `sleep` commands hang, time out, or produce garbled output with escape sequences.
- **Instead, wait locally between polls.** Use the local `terminal` tool with `sleep` to pause between checks, then send a fresh `send_command` to query state:

```sh
# Poll job state (fast command — no remote sleep)
sacct -j <jobid> -n -o State,Elapsed --noheader | head -1 && tail -5 slurm/logs/rollout_<jobid>.out

# Then wait locally (in terminal tool, not send_command):
sleep 180  # 3 minutes

# Then poll again:
sacct -j <jobid> -n -o State,Elapsed --noheader | head -1 && tail -5 slurm/logs/rollout_<jobid>.out
```

- Keep polling commands short and atomic — separate `sacct` and `tail` calls are fine.

## Training config models location

Training configuration Pydantic models (`TrainingRequestArgs`, `TrainingModeOneShot`,
`TrainProcessLaunchArgs`, etc.) live in **`src_py/training_config_models.py`** — a
standalone module **outside the `src_py/train/` package**. This is intentional:
importing from `src_py/train/*` triggers `src_py/train/__init__.py` which eagerly
imports `collator` → `import torch` → requires CUDA shared libraries.

The wrapper (`src_py/wrappers/training_wrapper.py`) imports directly from
`src_py.training_config_models`. The training engine (`src_py/train/main.py`) can
still import from `src_py.train.cli_args` (a re-export shim) since it already needs
torch.

When adding new training config models that the wrapper needs:
- Add them to `src_py/training_config_models.py` (not under `src_py/train/`)
- Re-export from `src_py/train/cli_args.py` if internal `train/` code uses them
- Keep imports torch-free in `training_config_models.py`

## Delta LD_LIBRARY_PATH

The `.slurm` scripts set up `LD_LIBRARY_PATH` to include nvidia CUDA libraries from
sglang's venv (excluding `nccl/lib`). When adding new `.slurm` scripts or modifying
the launch environment, ensure this pattern is replicated so that torch can find
`libcudnn.so.9`, `libcusparseLt.so.0`, etc. on compute nodes.
