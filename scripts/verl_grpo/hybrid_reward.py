"""Rule-based reward for isolated VERL GRPO runs on hybrid math data."""

from __future__ import annotations

import math
import re
from typing import Any

_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_HASH_RE = re.compile(r"####\s*(.+)")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d[\d,]*)?")


def _last_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1] if matches else None


def _after_hashes(text: str) -> str | None:
    matches = _HASH_RE.findall(text)
    return matches[-1] if matches else None


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = re.sub(r"^\\?text\s*\{(.+)\}$", r"\1", text).strip()
    return text


def _normalize(text: Any) -> str:
    value = _strip_wrappers(str(text))
    boxed = _last_boxed(value)
    if boxed is not None:
        value = boxed
    value = _strip_wrappers(value)
    value = value.lower()
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\u2212", "-")
    value = value.replace(",", "")
    value = re.sub(r"\\(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".。")
    return value


def _number_value(raw: str) -> float | None:
    text = _normalize(raw)
    if "/" in text and re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", text):
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return float(text)
    return None


def _candidate_answer(solution_str: str) -> str:
    hashed = _after_hashes(solution_str)
    if hashed is not None:
        return hashed.strip().splitlines()[0]
    boxed = _last_boxed(solution_str)
    if boxed is not None:
        return boxed
    numbers = _NUMBER_RE.findall(solution_str.replace(",", ""))
    if numbers:
        return numbers[-1]
    return solution_str.strip().splitlines()[-1] if solution_str.strip() else ""


def _equivalent(candidate: str, ground_truth: str) -> bool:
    candidate_norm = _normalize(candidate)
    truth_norm = _normalize(ground_truth)
    if candidate_norm and candidate_norm == truth_norm:
        return True
    candidate_num = _number_value(candidate_norm)
    truth_num = _number_value(truth_norm)
    if candidate_num is not None and truth_num is not None:
        return math.isclose(candidate_num, truth_num, rel_tol=1e-6, abs_tol=1e-6)
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    candidate = _candidate_answer(solution_str)
    if _equivalent(candidate, ground_truth):
        return 1.0
    if "####" in solution_str or _last_boxed(solution_str) is not None:
        return 0.1
    return 0.0
