# Agent Instructions

## Notification on job completion
After every task or job that you complete, call the Pushover notification script with a concise one-line summary of what was done:

```sh
python research-utility/scripts/pushover_notify.py "<one-line summary>"
```

Keep the message brief and descriptive — e.g. `"Fixed shape mismatch in attention layer"` or `"Ran full evaluation suite, all 42 tests pass"`.

## Delta SSH login workflow
When working with the `delta` host, prefer local SSH multiplexing over `ssh-tmux`.
Use `ssh-tmux` only as a fallback for interactive recovery when multiplexing cannot
be established.

Preferred multiplexed workflow:
1. Ensure `~/.ssh/config` for host `delta` enables `ControlMaster auto`,
   `ControlPersist`, and a stable `ControlPath`.
2. Start one master connection locally, for example:
   `ssh -MNf delta` after authentication is complete.
3. If authentication is interactive, have the user run `ssh delta` locally once
   and complete password + Duo there.
4. Confirm the control socket exists before running automated commands.
5. Reuse the authenticated transport with plain `ssh delta '<command>'` calls.
6. Close the master explicitly with `ssh -O exit delta` when finished.

Notes:
- Do not claim Delta access is unattended; a human must still complete MFA for the initial master connection.
- SSH multiplexing preserves the authenticated transport, not a shell working directory or shell-local environment.
- Never store passwords or MFA codes in repo files.
- Use `ssh-tmux` only if multiplexing is unavailable or broken.

## Delta multi-agent workflow
When multiple agents need concurrent Delta access, prefer a single multiplexed
master connection plus independent stateless `ssh delta '<command>'` invocations.

Recommended workflow:
1. Establish one authenticated master connection for host `delta`.
2. Let each agent issue its own direct SSH commands through that shared control socket.
3. Keep each command self-contained: explicit `cd`, explicit environment setup, explicit paths.
4. Use remote filesystem state, `sbatch`, `squeue`, and `sacct` for persistence rather than shared shell state.
5. If an agent truly needs an interactive shell, open a dedicated `ssh-tmux` session only for that agent.

Notes:
- Do not have multiple agents share one interactive shell.
- Multiplexing is appropriate for command execution; `ssh-tmux` is only the fallback for human-in-the-loop shell work.
- If the master connection dies, reauthenticate once and then resume ordinary SSH commands.

## Delta job monitoring
When monitoring long-running SLURM jobs on Delta:

- **Do NOT use `sleep` inside remote interactive shells.**
- **Instead, wait locally between polls.** Use the local terminal with `sleep` to pause between checks, then issue a fresh SSH command:

```sh
# Poll job state (fast command — no remote sleep)
ssh delta 'cd /u/zluo8/credit_assignment && sacct -j <jobid> -n -o State,Elapsed --noheader | head -1 && tail -5 slurm/logs/rollout_<jobid>.out'

# Then wait locally:
sleep 180  # 3 minutes

# Then poll again:
ssh delta 'cd /u/zluo8/credit_assignment && sacct -j <jobid> -n -o State,Elapsed --noheader | head -1 && tail -5 slurm/logs/rollout_<jobid>.out'
```

- Keep polling commands short and atomic — separate `sacct` and `tail` calls are fine.
- If no agent work is pending and no decision point needs user input, monitor
  relevant SLURM jobs at a 5-minute local interval until a decision point is
  reached.
- When a decision point is reached, push a concise notification and halt for
  user input.
- If a problem is obviously solvable without potential controversy, solve it,
  push a concise notification, and continue monitoring.

## Delta job submission reporting
When submitting SLURM jobs on Delta:

- Prefer the `scripts/hpc/*` launchers so resource requests such as GPU count
  and time limit are passed explicitly to `sbatch`.
- Do not rely on hardcoded `#SBATCH --time` fallbacks in reusable oneshot job
  scripts; pass `--time` explicitly through the launcher or direct `sbatch`
  command.
- After submission, report the requested resources to the user: job id, job
  name, account/QOS, partition, dependency if any, time limit, GPUs, CPUs,
  memory, and pending/running reason.
- Verify the submitted resource request with `scontrol show job <jobid>` or
  `squeue` rather than only trusting local config values.

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

Do not mix PyTorch/CUDA shared libraries across virtual environments on Delta.
If a job runs `torch` or `torchrun` from the root project `.venv`, any
`LD_LIBRARY_PATH` additions must come from that same `.venv`'s bundled
`site-packages/nvidia/*/lib` directories.

In particular, do not borrow cuDNN / cuSPARSELt / NCCL libraries from
`pyprojects/sglang/.venv` for root training jobs. That kind of cross-env mixing
causes brittle import failures and symbol mismatches such as
`undefined symbol: ncclCommResume`.

## LoRA rollout and generation naming

Rollout and training-set generation do not meaningfully differentiate between
LoRA and non-LoRA training variants. When preparing trajectories for both LoRA
and non-LoRA training, prefer the non-LoRA rollout and generation nicknames/configs
as the shared prerequisites.

In practice:
- LoRA and non-LoRA training jobs should both consume the non-LoRA rollout artifacts.
- LoRA and non-LoRA training jobs should both consume the non-LoRA generation artifacts.
- Do not launch or depend on a separate LoRA-specific rollout or generation config unless there is a new explicit reason in code or config semantics.
- If a LoRA training config points at a LoRA rollout or generation nickname, treat that as suspicious and verify whether it should instead reference the non-LoRA nickname.
