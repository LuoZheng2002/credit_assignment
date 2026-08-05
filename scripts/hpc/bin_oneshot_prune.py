#!/usr/bin/env python3
"""Submit a CPU SLURM job that prunes non-best one-shot checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_UTILITY_SRC = _REPO_ROOT / "research-utility" / "src_py"
if str(_RESEARCH_UTILITY_SRC) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_UTILITY_SRC))

from research_utility.slurm_submit import SlurmJobSpec, submit  # noqa: E402

SPEC = SlurmJobSpec(
    nickname_key="config_nickname_training",
    job_prefix="prune_",
    slurm_script_name="oneshot_prune_cpu.slurm",
    description="Submit a CPU SLURM job that prunes non-best one-shot checkpoints.",
    repo_root=_REPO_ROOT,
    request_gpu=False,
    partition="cpu",
    account="bfsl-delta-cpu",
)

if __name__ == "__main__":
    raise SystemExit(submit(SPEC))
