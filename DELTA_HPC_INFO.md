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
| Orchestrator     | `uv run python scripts/hpc/bin_orchestrator.py -c <config>`     | `config/orchestrator/qwen25_grpo_notool.toml`                   |
| Rollout (oneshot)| `uv run python scripts/hpc/bin_oneshot_rollout.py -c <config>`   | `config/oneshot_rollout/qwen25_rollout_grpo_notool.toml`        |
| Training (oneshot)| `uv run python scripts/hpc/bin_oneshot_training.py -c <config>`  | `config/oneshot_train/qwen25_train_grpo_notool.toml`            |

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

## Python Environment

**Always use `uv run`** to launch any Python script on Delta. The system `python` is 3.9
(lacks `tomllib`); `uv` provides Python 3.12 and manages dependencies via `pyproject.toml`.

```sh
uv run python scripts/hpc/bin_oneshot_rollout.py -c <config>
uv run python scripts/hpc/bin_oneshot_training.py -c <config>
```

### Virtual Environments

The project has two Python workspaces:

| Workspace | `pyproject.toml` location | Purpose |
|-----------|--------------------------|---------|
| Main | `pyproject.toml` (repo root) | Launcher scripts, data tools |
| SGLang | `pyprojects/sglang/pyproject.toml` | SGLang inference server (torch, flashinfer) |

To (re)initialize them:

```sh
uv sync                          # main workspace
uv sync --directory pyprojects/sglang  # sglang workspace
```

### CUDA Version & SGLang Compatibility

Delta HPC runs **CUDA 12.8** (driver 570.124.06). The sglang workspace MUST use the
`cu128` PyTorch index.

**SGLang version compatibility with CUDA 12.8:**

| sglang | `cuda-python` | `cutlass-dsl` | torch | CUDA 12.8? |
|--------|--------------|---------------|-------|------------|
| 0.5.10.post1 | `==12.9` | `>=4.4.1` | 2.9.1 | ✅ works |
| 0.5.11 | `>=13.0` | `==4.4.2` | 2.11.0 | ❌ cuda-python>=13 pulls cu13 bindings |
| 0.5.12.post1 | `>=13.0` | `[cu13]==4.5.1` | 2.11.0 | ❌ fully CUDA 13 only |

**sglang-kernel workaround:** sglang 0.5.10.post1 requires `sglang-kernel==0.4.1`,
but upstream only ships that as `+cu130`. The workspace uses a locally patched
wheel (0.4.2+cu129 re-versioned to 0.4.1) in `pyprojects/sglang/wheels/`.
If upgrading sglang in the future, re-check kernel compatibility.

### Disk Space Management

Home directory quota is tight (103G hard limit). The `uv` cache can grow large:

```sh
# Check home quota
quota -s

# Check uv cache size
du -sh ~/.cache/uv

# Clear it if low on space
uv cache clean
```

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

### Config Dependency Chain

**Rollout must finish before training.** The training binary looks up rollout trajectories
via `config_nickname_rollout`, so a matching rollout config must exist and its job must
complete first:

```sh
# Step 1: Launch rollout
uv run python scripts/hpc/bin_oneshot_rollout.py -c config/oneshot_rollout/qwen25_rollout_grpo_notool_lora.toml

# Step 2: After rollout completes, launch training
uv run python scripts/hpc/bin_oneshot_training.py -c config/oneshot_train/qwen25_train_grpo_notool_lora.toml
```

### Smoke Testing

For quick end-to-end tests, use the LoRA configs which are set to **1 GPU** with
`distributed_strategy = "single_gpu"`. Single-GPU jobs queue much faster than
multi-GPU:

```sh
# Smoke test rollout (1 GPU, ~300s)
uv run python scripts/hpc/bin_oneshot_rollout.py \
  -c config/oneshot_rollout/qwen25_rollout_grpo_notool_lora.toml

# Once rollout completes, smoke test training (1 GPU, ~10min/epoch)
uv run python scripts/hpc/bin_oneshot_training.py \
  -c config/oneshot_train/qwen25_train_grpo_notool_lora.toml
```

Check job status and logs:
```sh
squeue -u zluo8                                    # queue status
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed  # job outcome
cat slurm/logs/rollout_<jobid>.err                  # error log
cat slurm/logs/training_<jobid>.out                 # output log
```

### Available Launchable Configs

| Job Type | Config |
|----------|--------|
| Orchestrator | `config/orchestrator/qwen25_grpo_notool.toml` |
| Rollout | `config/oneshot_rollout/qwen25_rollout_grpo_notool.toml` |
| Rollout | `config/oneshot_rollout/qwen25_rollout_grpo_notool_lora.toml` |
| Training | `config/oneshot_train/qwen25_train_grpo_notool.toml` |
| Training | `config/oneshot_train/qwen25_train_grpo_notool_lora.toml` |

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

## Troubleshooting & Common Pitfalls

This section records issues encountered during setup and their solutions.
**Do not re-attempt dead-end approaches listed here.**

### 1. SLURM `$0` resolves to `/var/spool/slurmd/`

**Symptom:** `dirname "$0"` or `$0` in `.slurm` scripts resolves to `/var/spool/slurmd/jobXXXXX/`
instead of the repo.

**Cause:** `sbatch` copies the script to a spool directory before execution.

**Fix:** Use `$SLURM_SUBMIT_DIR` instead:
```sh
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
```
This is already applied to all `.slurm` files.

### 2. Python launcher ignores `#SBATCH` resource directives

**Symptom:** Jobs get only **1 CPU core** and **1 GB RAM** despite `.slurm` files
specifying `#SBATCH --cpus-per-task=32` and `#SBATCH --mem=32G`. Rust compilation
takes 10+ minutes instead of ~30s.

**Cause:** `research-utility/src_py/research_utility/slurm_submit.py` builds the
`sbatch` command with explicit CLI arguments, which override `#SBATCH` directives in
the script. The launcher was only passing `--gres`, `--time`, and `--account` —
missing `--cpus-per-task` and `--mem`.

**Fix (applied):** Added `--cpus-per-task 32` and `--mem 32G` (scaled by GPU count)
to the sbatch command in `slurm_submit.py`:
```python
"--cpus-per-task", str(cpus_per_task),
"--mem", f"{mem_per_gpu_gb * num_gpus}G",
```

**Verification:** Check with `scontrol show job <jobid> | grep -iE "cpus|mem"`.
Should show `NumCPUs=32` and `ReqTRES=...mem=32G...`.

### 3. Ninja not found during flashinfer JIT compilation

**Symptom:** SGLang server starts but crashes during CUDA graph capture:
```
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

**Cause:** flashinfer JIT-compiles CUDA kernels at runtime using `ninja` build
system. The `ninja` package is installed in the sglang venv, but the venv's
`bin/` directory is not on `PATH` when subprocesses are spawned.

**Fixes applied (both needed):**
1. Symlink ninja to a PATH-accessible location:
   ```sh
   ln -sf /u/zluo8/credit_assignment/pyprojects/sglang/.venv/bin/ninja ~/.local/bin/ninja
   ```
2. Add venv bin to PATH in all `.slurm` scripts so compute nodes find it:
   ```sh
   export PATH="$REPO_ROOT/pyprojects/sglang/.venv/bin:$PATH"
   ```
3. Added `ninja` to `pyprojects/sglang/pyproject.toml` dependencies for persistence.

### 4. `model_cli_name` must use CLI short names, not Rust variant names

**Symptom:** Panic at `LlmModelName::from_str`: `"invalid variant: qwen25_7b"`.

**Cause:** The `model_cli_name` field in TOML configs must match the
`CLI_NAME` constant on the model type (e.g., `Qwen25_7B::CLI_NAME = "qwen25"`),
not the Rust enum variant name (`Qwen25_7b`).

**Fix:** Use short CLI names:

| Wrong | Correct |
|-------|---------|
| `qwen25_7b` | `qwen25` |
| `qwen3_06b` | `qwen3-0.6b` |
| `qwen3_4b` | `qwen34` |
| `qwen35_08b` | `qwen3.5-0.8b` |
| `qwen35_4b` | `qwen3.5-4b` |
| `gemma3_4b` | `gemma` |
| `llama31_8b` | `llama` |
| `mistral7b_instruct_v03` | `mistral` |

All qwen25 configs have been corrected.

### 5. `TrainingSetSortMode` does not implement `Default`

**Symptom:** Rust compilation error:
```
error[E0277]: the trait bound `TrainingSetSortMode: Default` is not satisfied
  --> src/bin/bin_oneshot_rollout.rs:61:5
```

**Cause:** The rollout binary had `#[serde(default)]` on `training_set_sort_mode`,
which requires `Default`. But the field should always be explicitly specified.

**Fix (applied):** Removed `#[serde(default)]` from `training_set_sort_mode` in
`src/bin/bin_oneshot_rollout.rs`.

### 6. Mount / results directory must exist before job launch

**Symptom:** Job fails because the results directory doesn't exist.

**Fix:** Ensure the mount dir exists before submitting:
```sh
mkdir -p /work/nvme/bhph/zluo8/credit_assignment/results
```
The config sets `mount_dir = "/work/nvme/bhph/zluo8/credit_assignment/results"`.

### 7. SGLang version compatibility with CUDA 12.8

**Symptom:** sglang 0.5.11+ fails with CUDA 12.8. Only 0.5.10.post1 works.

**Cause:** sglang 0.5.11+ requires `cuda-python>=13.0` which pulls CUDA 13.x
bindings incompatible with Delta's CUDA 12.8 driver.

**Fix (applied):** Pin `sglang==v0.5.10.post1` in `pyprojects/sglang/pyproject.toml`.
See the CUDA compatibility table in the main SGLang section above.

**sglang-kernel patch:** sglang 0.5.10.post1 requires `sglang-kernel==0.4.1`,
but upstream only ships `0.4.1+cu130`. A locally patched wheel (0.4.2+cu129
re-versioned to 0.4.1) lives at `pyprojects/sglang/wheels/`. If `uv sync` is
re-run without this wheel, it will break.

### Quick Pre-flight Checklist

Before submitting any job, verify:

```sh
# 1. Mount dir exists
ls /work/nvme/bhph/zluo8/credit_assignment/results/

# 2. ninja is on PATH
which ninja

# 3. Both venvs initialized
uv run python -c "import tomllib; print('main ok')"
ls pyprojects/sglang/.venv/bin/python

# 4. Config model_cli_name is a short CLI name (e.g., "qwen25" not "qwen25_7b")
grep model_cli_name config/oneshot_rollout/qwen25_rollout_grpo_notool_lora.toml

# 5. Home quota has headroom (~10G+ free)
quota | grep "/u/zluo8"
```
