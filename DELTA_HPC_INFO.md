# Delta HPC Cluster Guide

## SSH Connection

The Delta cluster is configured in `~/.ssh/config` under the `delta` host alias:

```
Host delta
    HostName login.delta.ncsa.illinois.edu
    User zluo8
```

Delta requires **NCSA Kerberos + Duo MFA** — interactive authentication is mandatory.
This means agents **cannot** SSH in unattended; you must use a real terminal.

Connect with:

```sh
ssh delta
```

## Repository

- **Remote path:** `/u/zluo8/credit_assignment`
- **Work (NVMe) mount:** `/work/nvme/bhph/zluo8/credit_assignment/results` (change if disk fills up)

After SSHing in, always ensure the repo is up to date:

```sh
cd /u/zluo8/credit_assignment && git pull
```

Check disk quota with:

```sh
quota
```

## SLURM Account

All jobs use account: **`bfdz-delta-gpu`**

GPU type: **`nvidia_a100`** (request via `--gres=gpu:nvidia_a100:N`)

## Job Types & Launch Commands

Run these **from inside the Delta repo** (`/u/zluo8/credit_assignment`).
Each uses a TOML config file and calls `sbatch` under the hood:

| Job Type         | Script                                          | Example Config                                                  |
|------------------|-------------------------------------------------|-----------------------------------------------------------------|
| Orchestrator     | `python scripts/hpc/bin_orchestrator.py -c <config>`     | `config/orchestrator/qwen25_grpo_notool.toml`                   |
| Rollout (oneshot)| `python scripts/hpc/bin_oneshot_rollout.py -c <config>`   | `config/oneshot_rollout/qwen25_rollout_grpo_notool.toml`        |
| Training (oneshot)| `python scripts/hpc/bin_oneshot_training.py -c <config>`  | `config/oneshot_train/qwen25_train_grpo_notool.toml`            |

Optional: override the auto-generated job name with `-j <name>`.

## SLURM Scripts

The `.slurm` files live in `slurm/` and wrap `cargo run --release --bin <binary>`:

| Slurm Script                  | Binary                     |
|-------------------------------|----------------------------|
| `slurm/orchestrator.slurm`    | `bin_orchestrator`         |
| `slurm/oneshot_rollout.slurm` | `bin_oneshot_rollout`      |
| `slurm/oneshot_training.slurm`| `bin_oneshot_training`     |
| `slurm/gpu.slurm`             | (skeleton, sleeps forever) |
| `slurm/empty.slurm`           | (placeholder)              |

Logs land in `slurm/logs/`.

## Config Files

- **Top-level configs** (`config/`): rollout, training, and validation JSON configs
- **Orchestrator TOML:** `config/orchestrator/`
- **Rollout TOML:** `config/oneshot_rollout/`
- **Training TOML:** `config/oneshot_train/`

Each TOML config must contain these keys (validated by the launcher):
- `model_cli_name` (string)
- `config_nickname` / `config_nickname_rollout` / `config_nickname_training` (string, depending on job type)
- `num_gpus` (positive int)
- `total_time_limit_hours` (positive number — a 10% buffer is added automatically)

## Interactive Compute Node

To get a shell on a GPU node (for debugging or manual runs):

```sh
salloc --account=bfdz-delta-gpu --gres=gpu:nvidia_a100:1 --cpus-per-task=32 --mem=32G --time=02:00:00
```

## Persistent Sessions on Delta

Use `tmux` or `screen` on the login node to keep sessions alive across disconnects:

```sh
tmux new -s work
# ... do work ...
# Ctrl+B, D to detach
tmux attach -t work   # reattach later
```

## Constraints for Automated Agents

- **No unattended SSH** — Delta requires interactive MFA (Kerberos + Duo)
- **`sbatch` only exists on Delta** — launch scripts must run on the cluster, not locally
- **Agent `terminal` tool is stateless** — cannot maintain interactive SSH sessions
- **Workaround:** prepare configs/code locally, `git push`, then SSH in manually to pull and submit
