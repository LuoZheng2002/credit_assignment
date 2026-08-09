#!/usr/bin/env python3
"""Submit a SLURM job that runs bin_oneshot_ref_logprobs."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_UTILITY_SRC = _REPO_ROOT / "research-utility" / "src_py"
if str(_RESEARCH_UTILITY_SRC) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_UTILITY_SRC))

from research_utility.slurm_submit import SlurmJobSpec, submit  # noqa: E402

SPEC = SlurmJobSpec(
    nickname_key="config_nickname_generation",
    job_prefix="ref_logprobs_",
    slurm_script_name="oneshot_ref_logprobs.slurm",
    description="Submit a SLURM job that runs bin_oneshot_ref_logprobs.",
    repo_root=_REPO_ROOT,
)

if __name__ == "__main__":
    raise SystemExit(submit(SPEC))
