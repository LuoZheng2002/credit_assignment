from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from _bootstrap import REPO_ROOT
from dotenv import load_dotenv

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/v1/chat/completions"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

DEEPSEEK_V4_FLASH_MODEL = "deepseek-v4-flash"
DEEPSEEK_V4_FLASH_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"

JUDGE_TOTAL_ATTEMPTS = 10
GENERATION_TOTAL_ATTEMPTS = 5
REQUEST_TIMEOUT_SECS = 300.0
DEFAULT_MAX_CONCURRENCY = 200
DEFAULT_INPUT_LIMIT = 10_000
DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_MAX_ACCEPTED_SAMPLES = 2_000

GENERATION_KIND_OPENAI = "openai"
GENERATION_KIND_DEEPSEEK_OFFICIAL = "deepseek_official"
JUDGE_KIND_OPENROUTER = "openrouter"
JUDGE_KIND_DEEPSEEK_OFFICIAL = "deepseek_official"


@dataclass(frozen=True)
class GenerationBackendConfig:
    kind: str
    api_url: str
    default_model: str
    api_key_env: str
    description_label: str


@dataclass(frozen=True)
class JudgeBackendConfig:
    kind: str
    api_url: str
    model: str
    api_key_env: str
    description_label: str


@dataclass(frozen=True)
class ProgramDefaults:
    description: str
    generation_backend: GenerationBackendConfig
    judge_backend: JudgeBackendConfig
    default_output: Path
    default_rejected_output: Path
    default_progress_path: Path


@dataclass(frozen=True)
class ProgramArgs:
    input: Path
    output: Path
    rejected_output: Path
    progress_path: Path
    limit: int
    max_completion_tokens: int
    generation_temperature: float
    max_concurrency: int
    max_accepted_samples: int
    overwrite: bool
    generation_model: str


@dataclass(frozen=True)
class HybridTrainEntry:
    flat_id: int
    dataset_name: str
    question_id: int
    question: str
    correct_answer: str


@dataclass(frozen=True)
class SftAcceptedEntry:
    flat_id: int
    dataset_name: str
    question_id: int
    question: str
    correct_answer: str
    prompt: str
    reference_trajectory: str


@dataclass(frozen=True)
class GenerationResult:
    trajectory: str
    model_answer: str | None
    is_correct: bool


@dataclass(frozen=True)
class RejectedEntry:
    flat_id: int
    dataset_name: str
    question_id: int
    question: str
    correct_answer: str
    prompt: str
    reference_trajectory: str
    model_answer: str | None
    is_correct: bool
    error: str | None


@dataclass(frozen=True)
class ProgressState:
    started_at_unix: float
    last_update_unix: float
    input_path: str
    accepted_output_path: str
    rejected_output_path: str
    progress_path: str
    total_considered: int
    max_concurrency: int
    max_completion_tokens: int
    max_accepted_samples: int
    generation_temperature: float
    generation_model: str
    generation_backend: str
    judge_backend: str
    processed_count: int
    accepted_count: int
    rejected_count: int
    skipped_already_processed_count: int
    inflight_count: int
    last_completed_flat_id: int | None


def prompt_without_tool_call(question: str) -> str:
    return (
        "You are a helpful agent that solves the following problem.\n"
        f"Question: {question}\n"
        "You should reason step by step, and put the final answer in a \\boxed{}.\n"
        "Begin your reasoning:"
    )


def extract_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    search_start = 0

    while True:
        relative_start = text[search_start:].find(marker)
        if relative_start < 0:
            return None
        start = search_start + relative_start
        after_marker = start + len(marker)

        bracket_depth = 1
        content_chars: list[str] = []
        end_index_after_closing_brace: int | None = None

        for offset, ch in enumerate(text[after_marker:]):
            if ch == "{":
                bracket_depth += 1
                content_chars.append(ch)
            elif ch == "}":
                bracket_depth -= 1
                if bracket_depth == 0:
                    end_index_after_closing_brace = after_marker + offset + 1
                    break
                content_chars.append(ch)
            else:
                content_chars.append(ch)

        if end_index_after_closing_brace is None:
            return None

        content = "".join(content_chars)
        if content.strip():
            return content

        search_start = end_index_after_closing_brace


def extract_boxed_verdict(response: str) -> str | None:
    marker = "\\boxed{"
    start = response.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    remaining = response[content_start:]
    depth = 1
    for i, ch in enumerate(remaining):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return remaining[:i].strip()
    return None


def _load_env() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


def _build_parser(defaults: ProgramDefaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=defaults.description)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "hybrid_train.jsonl",
        help="Input hybrid train JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.default_output,
        help="Accepted SFT JSONL path",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=defaults.default_rejected_output,
        help="Rejected trajectory audit JSONL path",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=defaults.default_progress_path,
        help="Progress metadata JSON path used for crash-safe resume",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_INPUT_LIMIT,
        help="Number of rows from the front of hybrid_train.jsonl to consider",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Maximum completion tokens for generation",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=1.0,
        help="Temperature used for generation",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum number of concurrent generation+judge workers",
    )
    parser.add_argument(
        "--max-accepted-samples",
        type=int,
        default=DEFAULT_MAX_ACCEPTED_SAMPLES,
        help="Stop once this many accepted SFT rows have been written",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=defaults.generation_backend.default_model,
        help="Generation model name to send to the provider API",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing accepted/rejected/progress outputs before generating",
    )
    return parser


def _parse_args(defaults: ProgramDefaults) -> ProgramArgs:
    parser = _build_parser(defaults)
    ns = parser.parse_args()
    assert ns.limit > 0, "--limit must be positive"
    assert ns.max_completion_tokens > 0, "--max-completion-tokens must be positive"
    assert ns.max_concurrency > 0, "--max-concurrency must be positive"
    assert ns.max_accepted_samples > 0, "--max-accepted-samples must be positive"
    assert ns.generation_model.strip(), "--generation-model must be non-empty"
    return ProgramArgs(
        input=ns.input,
        output=ns.output,
        rejected_output=ns.rejected_output,
        progress_path=ns.progress_path,
        limit=ns.limit,
        max_completion_tokens=ns.max_completion_tokens,
        generation_temperature=ns.generation_temperature,
        max_concurrency=ns.max_concurrency,
        max_accepted_samples=ns.max_accepted_samples,
        overwrite=bool(ns.overwrite),
        generation_model=str(ns.generation_model).strip(),
    )


def _read_jsonl(path: Path, limit: int) -> list[HybridTrainEntry]:
    assert path.is_file(), f"input dataset does not exist: {path}"
    rows: list[HybridTrainEntry] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if len(rows) >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                HybridTrainEntry(
                    flat_id=int(payload["flat_id"]),
                    dataset_name=str(payload["dataset_name"]),
                    question_id=int(payload["question_id"]),
                    question=str(payload["question"]),
                    correct_answer=str(payload["correct_answer"]),
                )
            )
    assert rows, f"no rows loaded from {path}"
    return rows


def _iter_jsonl_objects_tolerant(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as err:
                print(
                    f"Warning: skipping malformed JSONL line in {path} at line {line_number}: {err}"
                )
                continue
            if not isinstance(payload, dict):
                print(
                    f"Warning: skipping non-object JSONL line in {path} at line {line_number}"
                )
                continue
            payloads.append(payload)
    return payloads


def _read_processed_flat_ids(paths: list[Path]) -> set[int]:
    processed: set[int] = set()
    for path in paths:
        for payload in _iter_jsonl_objects_tolerant(path):
            processed.add(int(payload["flat_id"]))
    return processed


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temp_path.replace(path)


def _count_jsonl_rows(path: Path) -> int:
    return len(_iter_jsonl_objects_tolerant(path))


def _read_existing_started_at(progress_path: Path) -> float | None:
    if not progress_path.exists():
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as err:
        print(f"Warning: failed to read existing progress file {progress_path}: {err}")
        return None
    if not isinstance(payload, dict):
        return None
    started_at = payload.get("started_at_unix")
    if isinstance(started_at, (int, float)):
        return float(started_at)
    return None


def _save_progress(
    path: Path,
    *,
    defaults: ProgramDefaults,
    args: ProgramArgs,
    started_at_unix: float,
    total_considered: int,
    processed_count: int,
    accepted_count: int,
    rejected_count: int,
    skipped_already_processed_count: int,
    inflight_count: int,
    last_completed_flat_id: int | None,
) -> None:
    state = ProgressState(
        started_at_unix=started_at_unix,
        last_update_unix=time.time(),
        input_path=str(args.input),
        accepted_output_path=str(args.output),
        rejected_output_path=str(args.rejected_output),
        progress_path=str(path),
        total_considered=total_considered,
        max_concurrency=args.max_concurrency,
        max_completion_tokens=args.max_completion_tokens,
        max_accepted_samples=args.max_accepted_samples,
        generation_temperature=args.generation_temperature,
        generation_model=args.generation_model,
        generation_backend=defaults.generation_backend.description_label,
        judge_backend=defaults.judge_backend.description_label,
        processed_count=processed_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        skipped_already_processed_count=skipped_already_processed_count,
        inflight_count=inflight_count,
        last_completed_flat_id=last_completed_flat_id,
    )
    _write_json_atomic(path, asdict(state))


def _extract_message_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for entry in message_content:
            if isinstance(entry, dict):
                text = entry.get("text")
                if isinstance(text, str):
                    parts.append(text)
        merged = "".join(parts)
        if merged:
            return merged
    raise ValueError(f"message content has unexpected shape: {message_content!r}")


def _required_api_key(env_name: str) -> str:
    api_key = os.environ.get(env_name, "").strip()
    assert api_key, f"{env_name} environment variable not set"
    return api_key


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(url, headers=headers, json=body)
    raw_text = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.status_code >= 400:
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = payload["error"].get("message")
            if isinstance(message, str) and message.strip():
                raise RuntimeError(message)
        raise RuntimeError(
            f"HTTP {response.status_code} calling {url}: {raw_text[:2000]}"
        )

    if isinstance(payload, dict):
        if isinstance(payload.get("error"), dict):
            message = payload["error"].get("message")
            if isinstance(message, str) and message.strip():
                raise RuntimeError(message)
        return payload

    raise RuntimeError(
        f"Expected JSON object response from {url}, got: {raw_text[:2000]}"
    )


async def _generate_trajectory(
    client: httpx.AsyncClient,
    generation_backend: GenerationBackendConfig,
    *,
    generation_model: str,
    prompt: str,
    max_completion_tokens: int,
    temperature: float,
) -> str:
    api_key = _required_api_key(generation_backend.api_key_env)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_body: dict[str, Any] = {
        "model": generation_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    last_error: Exception | None = None
    for attempt in range(GENERATION_TOTAL_ATTEMPTS):
        try:
            body = dict(base_body)
            if generation_backend.kind == GENERATION_KIND_OPENAI:
                body["max_completion_tokens"] = max_completion_tokens
            elif generation_backend.kind == GENERATION_KIND_DEEPSEEK_OFFICIAL:
                body["max_completion_tokens"] = max_completion_tokens
                body["thinking"] = {"type": "disabled"}
            else:
                raise AssertionError(
                    f"unsupported generation backend kind: {generation_backend.kind}"
                )
            payload = await _post_json(
                client,
                generation_backend.api_url,
                headers,
                body,
            )
            return _extract_message_content(payload["choices"][0]["message"]["content"])
        except Exception as err:
            last_error = err
            err_text = str(err)
            retry_with_max_tokens = generation_backend.kind in {
                GENERATION_KIND_OPENAI,
                GENERATION_KIND_DEEPSEEK_OFFICIAL,
            } and (
                "max_completion_tokens" in err_text
                or "max_tokens" in err_text
                or "Unrecognized request argument" in err_text
                or "Unknown parameter" in err_text
            )
            if retry_with_max_tokens:
                try:
                    body = dict(base_body)
                    body["max_tokens"] = max_completion_tokens
                    payload = await _post_json(
                        client,
                        generation_backend.api_url,
                        headers,
                        body,
                    )
                    return _extract_message_content(
                        payload["choices"][0]["message"]["content"]
                    )
                except Exception as fallback_err:
                    last_error = fallback_err

        if attempt + 1 < GENERATION_TOTAL_ATTEMPTS:
            await asyncio.sleep(float(attempt + 1))

    raise RuntimeError(
        f"failed to generate trajectory after {GENERATION_TOTAL_ATTEMPTS} attempts: {last_error}"
    )


async def _fetch_judge_evaluation(
    client: httpx.AsyncClient,
    judge_backend: JudgeBackendConfig,
    *,
    prompt: str,
    temperature: float,
    thinking_enabled: bool,
) -> tuple[str, str | None]:
    api_key = _required_api_key(judge_backend.api_key_env)
    body: dict[str, Any] = {
        "model": judge_backend.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 4096,
        "temperature": temperature,
    }
    if judge_backend.kind == JUDGE_KIND_OPENROUTER:
        body["reasoning"] = {
            "effort": "high" if thinking_enabled else "none",
        }
    elif judge_backend.kind == JUDGE_KIND_DEEPSEEK_OFFICIAL:
        body["thinking"] = {
            "type": "enabled" if thinking_enabled else "disabled",
        }
    else:
        raise AssertionError(f"unsupported judge backend kind: {judge_backend.kind}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/luoz/credit_assignment",
        "X-Title": "credit_assignment",
    }
    payload = await _post_json(client, judge_backend.api_url, headers, body)
    message = payload["choices"][0]["message"]
    reasoning_value = None
    if isinstance(message, dict):
        raw_reasoning = message.get("reasoning_content")
        if not isinstance(raw_reasoning, str) or not raw_reasoning.strip():
            raw_reasoning = message.get("reasoning")
        if isinstance(raw_reasoning, str) and raw_reasoning.strip():
            reasoning_value = raw_reasoning.strip()
    return _extract_message_content(message["content"]), reasoning_value


async def _judge_answer(
    client: httpx.AsyncClient,
    judge_backend: JudgeBackendConfig,
    *,
    model_answer: str,
    correct_answer: str,
    question: str,
) -> bool:
    base_prompt = (
        "You are an answer checker that checks a model's answer against the reference answer. "
        "Judge if the model's answer is equivalent to the reference answer. "
        "Do not attempt to solve the problem yourself, only judge whether the given answer and "
        "the reference answer is equivalent. "
        "If the model's answer contains units but the reference answer does not, treat them as "
        "equivalent if the numerical values are the same. \n"
        f'The question is: "{question}". '
        f'The model\'s answer is: "{model_answer}", and the correct answer is: "{correct_answer}".'
    )
    thinking_prompt = (
        f"{base_prompt} Think step by step about whether the model's answer matches the "
        "reference answer. Put your final answer in \\boxed{correct} or \\boxed{incorrect}."
    )

    last_error: str | None = None
    for attempt in range(JUDGE_TOTAL_ATTEMPTS):
        temperature = 0.0 if attempt == 0 else 1.0
        thinking_enabled = False
        try:
            evaluation, _reasoning = await _fetch_judge_evaluation(
                client,
                judge_backend,
                prompt=thinking_prompt,
                temperature=temperature,
                thinking_enabled=thinking_enabled,
            )
            verdict = extract_boxed_verdict(evaluation)
            if verdict is None:
                last_error = (
                    f"No \\boxed{{}} found in evaluation response: {evaluation}"
                )
            else:
                verdict_lower = verdict.lower()
                if "incorrect" in verdict_lower:
                    return False
                if "correct" in verdict_lower:
                    return True
                last_error = (
                    "Verdict in \\boxed{} was neither 'correct' nor 'incorrect': "
                    f"{verdict}"
                )
        except Exception as err:
            last_error = str(err)

        if attempt + 1 < JUDGE_TOTAL_ATTEMPTS:
            print(
                "Judger returned invalid response, "
                f"attempt {attempt + 1}/{JUDGE_TOTAL_ATTEMPTS}. Last error: {last_error}"
            )
            await asyncio.sleep(1.0)

    raise RuntimeError(
        f"Failed to judge answer after {JUDGE_TOTAL_ATTEMPTS} attempts: {last_error}"
    )


async def _process_entry(
    client: httpx.AsyncClient,
    generation_backend: GenerationBackendConfig,
    judge_backend: JudgeBackendConfig,
    args: ProgramArgs,
    entry: HybridTrainEntry,
) -> tuple[HybridTrainEntry, str, GenerationResult]:
    prompt = prompt_without_tool_call(entry.question)
    trajectory = await _generate_trajectory(
        client,
        generation_backend,
        generation_model=args.generation_model,
        prompt=prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.generation_temperature,
    )
    model_answer = extract_boxed_content(trajectory)
    if model_answer is None:
        return (
            entry,
            prompt,
            GenerationResult(
                trajectory=trajectory,
                model_answer=None,
                is_correct=False,
            ),
        )
    is_correct = await _judge_answer(
        client,
        judge_backend,
        model_answer=model_answer,
        correct_answer=entry.correct_answer,
        question=entry.question,
    )
    return (
        entry,
        prompt,
        GenerationResult(
            trajectory=trajectory,
            model_answer=model_answer,
            is_correct=is_correct,
        ),
    )


async def _run_async(defaults: ProgramDefaults, args: ProgramArgs) -> None:
    if args.overwrite:
        for path in (args.output, args.rejected_output, args.progress_path):
            if path.exists():
                path.unlink()

    rows = _read_jsonl(args.input, limit=args.limit)
    processed_flat_ids = _read_processed_flat_ids([args.output, args.rejected_output])
    pending_rows = [row for row in rows if row.flat_id not in processed_flat_ids]
    skipped_already_processed_count = len(rows) - len(pending_rows)

    accepted_count = _count_jsonl_rows(args.output)
    rejected_count = _count_jsonl_rows(args.rejected_output)
    processed_count = accepted_count + rejected_count
    initial_processed_count = processed_count
    started_at_unix = _read_existing_started_at(args.progress_path) or time.time()

    _save_progress(
        args.progress_path,
        defaults=defaults,
        args=args,
        started_at_unix=started_at_unix,
        total_considered=len(rows),
        processed_count=processed_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        skipped_already_processed_count=skipped_already_processed_count,
        inflight_count=0,
        last_completed_flat_id=None,
    )

    print(
        f"Loaded {len(rows)} candidate rows from {args.input}. "
        f"Pending after resume filter: {len(pending_rows)}. "
        f"Already processed: {skipped_already_processed_count}."
    )
    print(
        f"Generation backend: {defaults.generation_backend.description_label} model={args.generation_model}\n"
        f"Judge backend: {defaults.judge_backend.description_label} model={defaults.judge_backend.model}\n"
        f"Accepted output: {args.output}\n"
        f"Rejected output: {args.rejected_output}\n"
        f"Progress path: {args.progress_path}\n"
        f"Max concurrency: {args.max_concurrency}\n"
        f"Accepted sample cap: {args.max_accepted_samples}"
    )

    if accepted_count >= args.max_accepted_samples:
        print(
            f"Accepted output already contains {accepted_count} rows, which meets/exceeds the cap "
            f"of {args.max_accepted_samples}. Nothing to do."
        )
        return

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECS)
    limits = httpx.Limits(
        max_connections=max(args.max_concurrency * 2, 20),
        max_keepalive_connections=max(args.max_concurrency * 2, 20),
    )

    pending_index = 0
    last_completed_flat_id: int | None = None
    cap_reached = False

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        in_flight: dict[
            asyncio.Task[tuple[HybridTrainEntry, str, GenerationResult]],
            HybridTrainEntry,
        ] = {}

        def schedule_more() -> None:
            nonlocal pending_index
            while (
                pending_index < len(pending_rows)
                and len(in_flight) < args.max_concurrency
                and accepted_count < args.max_accepted_samples
            ):
                row = pending_rows[pending_index]
                pending_index += 1
                task = asyncio.create_task(
                    _process_entry(
                        client,
                        defaults.generation_backend,
                        defaults.judge_backend,
                        args,
                        row,
                    )
                )
                in_flight[task] = row

        schedule_more()

        while in_flight:
            done, pending = await asyncio.wait(
                in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                source_row = in_flight.pop(task)
                try:
                    row, prompt, result = await task
                except asyncio.CancelledError:
                    continue
                except Exception as err:
                    prompt = prompt_without_tool_call(source_row.question)
                    result_error = RejectedEntry(
                        flat_id=source_row.flat_id,
                        dataset_name=source_row.dataset_name,
                        question_id=source_row.question_id,
                        question=source_row.question,
                        correct_answer=source_row.correct_answer,
                        prompt=prompt,
                        reference_trajectory="",
                        model_answer=None,
                        is_correct=False,
                        error=str(err),
                    )
                    _append_jsonl(args.rejected_output, asdict(result_error))
                    rejected_count += 1
                    processed_count += 1
                    processed_flat_ids.add(source_row.flat_id)
                    last_completed_flat_id = source_row.flat_id
                    print(
                        f"rejected flat_id={source_row.flat_id} due to worker error: {err} "
                        f"processed={processed_count}/{len(rows)}"
                    )
                else:
                    if result.is_correct and accepted_count < args.max_accepted_samples:
                        accepted = SftAcceptedEntry(
                            flat_id=row.flat_id,
                            dataset_name=row.dataset_name,
                            question_id=row.question_id,
                            question=row.question,
                            correct_answer=row.correct_answer,
                            prompt=prompt,
                            reference_trajectory=result.trajectory,
                        )
                        _append_jsonl(args.output, asdict(accepted))
                        accepted_count += 1
                        processed_count += 1
                        processed_flat_ids.add(row.flat_id)
                        last_completed_flat_id = row.flat_id
                        print(
                            f"accepted flat_id={row.flat_id} model_answer={result.model_answer!r} "
                            f"processed={processed_count}/{len(rows)} accepted={accepted_count}"
                        )
                    else:
                        rejected = RejectedEntry(
                            flat_id=row.flat_id,
                            dataset_name=row.dataset_name,
                            question_id=row.question_id,
                            question=row.question,
                            correct_answer=row.correct_answer,
                            prompt=prompt,
                            reference_trajectory=result.trajectory,
                            model_answer=result.model_answer,
                            is_correct=False,
                            error=(
                                "accepted sample cap reached"
                                if result.is_correct
                                and accepted_count >= args.max_accepted_samples
                                else None
                            ),
                        )
                        _append_jsonl(args.rejected_output, asdict(rejected))
                        rejected_count += 1
                        processed_count += 1
                        processed_flat_ids.add(row.flat_id)
                        last_completed_flat_id = row.flat_id
                        print(
                            f"rejected flat_id={row.flat_id} model_answer={result.model_answer!r} "
                            f"processed={processed_count}/{len(rows)} rejected={rejected_count}"
                        )

                if accepted_count >= args.max_accepted_samples:
                    cap_reached = True

            if cap_reached and pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                in_flight.clear()
                print(
                    f"Accepted cap {args.max_accepted_samples} reached; cancelled {len(pending)} in-flight tasks."
                )
            elif not cap_reached:
                schedule_more()

            _save_progress(
                args.progress_path,
                defaults=defaults,
                args=args,
                started_at_unix=started_at_unix,
                total_considered=len(rows),
                processed_count=processed_count,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                skipped_already_processed_count=skipped_already_processed_count,
                inflight_count=len(in_flight),
                last_completed_flat_id=last_completed_flat_id,
            )

    _save_progress(
        args.progress_path,
        defaults=defaults,
        args=args,
        started_at_unix=started_at_unix,
        total_considered=len(rows),
        processed_count=processed_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        skipped_already_processed_count=skipped_already_processed_count,
        inflight_count=0,
        last_completed_flat_id=last_completed_flat_id,
    )

    print()
    print("Generation finished.")
    print(f"  considered: {len(rows)}")
    print(f"  skipped_already_processed: {skipped_already_processed_count}")
    print(f"  processed_this_run: {processed_count - initial_processed_count}")
    print(f"  accepted_total: {accepted_count}")
    print(f"  accepted_sample_cap: {args.max_accepted_samples}")
    print(f"  rejected_total: {rejected_count}")
    print(f"  accepted_output: {args.output}")
    print(f"  rejected_output: {args.rejected_output}")
    print(f"  progress_path: {args.progress_path}")


def run_generation_program(defaults: ProgramDefaults) -> None:
    _load_env()
    args = _parse_args(defaults)
    asyncio.run(_run_async(defaults, args))
