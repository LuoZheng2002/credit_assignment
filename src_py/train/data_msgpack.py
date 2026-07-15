from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator

import msgpack


@dataclass(frozen=True)
class QuestionNodeId:
    question_id: int
    node_id: int


@dataclass(frozen=True)
class TrainingSampleTokenized:
    id: QuestionNodeId
    input_ids: list[int]
    labels: list[int]
    input_length: int
    token_advantages: list[float]
    model_official_name: str


def _assert_msgpack_path_exists(msgpack_path: str) -> None:
    assert os.path.isfile(msgpack_path), f"msgpack file not found: {msgpack_path}"


def _parse_payload_object(payload_obj: object, payload_kind: str) -> dict[str, object]:
    assert isinstance(payload_obj, dict), f"{payload_kind} payload must be a dictionary"
    return payload_obj


def _parse_question_node_id(value: object) -> QuestionNodeId:
    assert isinstance(value, dict), "QuestionNodeId must be an object"
    assert "question_id" in value, "QuestionNodeId must contain question_id"
    assert "node_id" in value, "QuestionNodeId must contain node_id"

    question_id_obj = value["question_id"]
    node_id_obj = value["node_id"]
    assert isinstance(question_id_obj, int), "question_id must be int"
    assert isinstance(node_id_obj, int), "node_id must be int"
    assert question_id_obj >= 0, "question_id must be non-negative"
    assert node_id_obj >= 0, "node_id must be non-negative"

    return QuestionNodeId(question_id=question_id_obj, node_id=node_id_obj)


def _parse_int_list(value: object, field_name: str) -> list[int]:
    assert isinstance(value, list), f"{field_name} must be a list"
    output: list[int] = []
    for element in value:
        assert isinstance(element, int), f"{field_name} must contain int values"
        output.append(element)
    return output


def _parse_finite_float(value: object, field_name: str) -> float:
    assert isinstance(value, (float, int)), f"{field_name} must be numeric"
    numeric_value = float(value)
    assert math.isfinite(numeric_value), f"{field_name} must be finite"
    return numeric_value


def _parse_positive_int(value: object, field_name: str) -> int:
    assert isinstance(value, int), f"{field_name} must be int"
    assert value >= 0, f"{field_name} must be non-negative"
    return value


def _parse_tokenized_payload(payload: object) -> TrainingSampleTokenized:
    payload_obj = _parse_payload_object(payload, "tokenized")
    assert "id" in payload_obj, "tokenized payload must contain id"
    assert "input_ids" in payload_obj, "tokenized payload must contain input_ids"
    assert "labels" in payload_obj, "tokenized payload must contain labels"
    assert "input_length" in payload_obj, "tokenized payload must contain input_length"
    assert "token_advantages" in payload_obj or "advantage" in payload_obj, (
        "tokenized payload must contain token_advantages or advantage"
    )
    assert "model_official_name" in payload_obj, (
        "tokenized payload must contain model_official_name"
    )

    sample_id = _parse_question_node_id(payload_obj["id"])
    input_ids = _parse_int_list(payload_obj["input_ids"], "input_ids")
    labels = _parse_int_list(payload_obj["labels"], "labels")
    input_length = _parse_positive_int(payload_obj["input_length"], "input_length")
    model_official_name_obj = payload_obj["model_official_name"]
    if "token_advantages" in payload_obj:
        token_advantages_obj = payload_obj["token_advantages"]
        assert isinstance(token_advantages_obj, list), "token_advantages must be a list"
        token_advantages = []
        for element in token_advantages_obj:
            token_advantages.append(_parse_finite_float(element, "token_advantages[]"))
    else:
        advantage = _parse_finite_float(payload_obj["advantage"], "advantage")
        token_advantages = [advantage] * input_length
    assert isinstance(model_official_name_obj, str), (
        "model_official_name must be string"
    )
    assert len(model_official_name_obj) > 0, "model_official_name cannot be empty"

    assert len(input_ids) > 0, "input_ids cannot be empty"
    assert len(labels) == len(input_ids), "labels and input_ids lengths must match"
    assert input_length == len(input_ids), "input_length must equal len(input_ids)"
    assert len(token_advantages) == input_length, (
        "token_advantages and input_ids lengths must match"
    )

    return TrainingSampleTokenized(
        id=sample_id,
        input_ids=input_ids,
        labels=labels,
        input_length=input_length,
        token_advantages=token_advantages,
        model_official_name=model_official_name_obj,
    )


def _parse_direct_training_trajectory_payload(
    trajectory_id: int,
    payload: object,
) -> TrainingSampleTokenized:
    payload_obj = _parse_payload_object(payload, "trajectory")
    if (
        "input_ids" not in payload_obj
        or "labels" not in payload_obj
        or "advantages" not in payload_obj
    ):
        if "question" in payload_obj and "tree" in payload_obj:
            raise AssertionError(
                "trajectory payload appears to be DirectTreeActionLog; expected DirectTrainingTrajectory "
                "with input_ids/labels/advantages"
            )
    assert "question" in payload_obj, "trajectory payload must contain question"
    assert "input_ids" in payload_obj, "trajectory payload must contain input_ids"
    assert "labels" in payload_obj, "trajectory payload must contain labels"
    assert "advantages" in payload_obj, "trajectory payload must contain advantages"

    question_obj = payload_obj["question"]
    assert isinstance(question_obj, dict), "question must be an object"
    assert "flat_id" in question_obj, "question must contain flat_id"
    question_flat_id = _parse_positive_int(question_obj["flat_id"], "question.flat_id")

    input_ids = _parse_int_list(payload_obj["input_ids"], "input_ids")
    labels = _parse_int_list(payload_obj["labels"], "labels")

    advantages_obj = payload_obj["advantages"]
    assert isinstance(advantages_obj, list), "advantages must be a list"
    token_advantages: list[float] = []
    for element in advantages_obj:
        token_advantages.append(_parse_finite_float(element, "advantages[]"))

    assert len(input_ids) > 0, "input_ids cannot be empty"
    assert len(labels) == len(input_ids), "labels and input_ids lengths must match"
    assert len(token_advantages) == len(input_ids), (
        "advantages and input_ids lengths must match"
    )

    assert len(input_ids) >= 2, "trajectory must contain at least two tokens"

    supervised_advantages = [
        token_advantages[index] for index, label in enumerate(labels) if label != -100
    ]
    assert len(supervised_advantages) > 0, (
        "trajectory must contain at least one supervised token"
    )
    assert any(label != -100 for label in labels[1:]), (
        "trajectory must contain at least one supervised token after causal shift"
    )

    return TrainingSampleTokenized(
        id=QuestionNodeId(question_id=question_flat_id, node_id=trajectory_id),
        input_ids=input_ids,
        labels=labels,
        input_length=len(input_ids),
        token_advantages=token_advantages,
        model_official_name="",
    )


def _iter_msgpack_payloads(msgpack_path: str) -> Iterator[object]:
    _assert_msgpack_path_exists(msgpack_path)
    with open(msgpack_path, "rb") as file:
        unpacker = msgpack.Unpacker(file, raw=False)
        for payload in unpacker:
            yield payload


class LazyTrainingTrajectoryStore:
    def __init__(
        self,
        msgpack_path: str,
        first_n_training_samples: int = 0,
        training_trajectory_len_cutoff: int = 4096,
    ):
        _assert_msgpack_path_exists(msgpack_path)
        assert first_n_training_samples >= 0, (
            "first_n_training_samples must be non-negative"
        )
        assert training_trajectory_len_cutoff >= 2, (
            "training_trajectory_len_cutoff must be at least 2"
        )

        self._msgpack_path = msgpack_path
        self._offsets: list[int] = []
        previous_input_length = -1
        with open(msgpack_path, "rb") as file:
            unpacker = msgpack.Unpacker(file, raw=False)
            start_offset = 0
            for trajectory_id, payload in enumerate(unpacker):
                sample = _parse_direct_training_trajectory_payload(trajectory_id, payload)
                assert (
                    sample.input_length <= previous_input_length or previous_input_length == -1
                ), (
                    "training trajectories must be sorted by input_ids length in descending order"
                )
                previous_input_length = sample.input_length
                if sample.input_length <= training_trajectory_len_cutoff:
                    self._offsets.append(start_offset)
                    if (
                        first_n_training_samples > 0
                        and len(self._offsets) == first_n_training_samples
                    ):
                        break
                start_offset = unpacker.tell()
        self._sample_count = len(self._offsets)
        assert self._sample_count > 0, "training trajectory data must be non-empty"
        self.sample_count = self._sample_count

    def close(self) -> None:
        return None

    def _load_payload_at_offset(self, offset: int) -> object:
        with open(self._msgpack_path, "rb") as file:
            file.seek(offset)
            unpacker = msgpack.Unpacker(file, raw=False)
            try:
                return next(unpacker)
            except StopIteration as error:
                raise AssertionError(
                    f"trajectory index must be contiguous: missing payload at byte offset {offset}"
                ) from error

    def get_sample(self, trajectory_id: int) -> TrainingSampleTokenized:
        assert trajectory_id >= 0, "trajectory_id must be non-negative"
        assert trajectory_id < self.sample_count, "trajectory_id out of bounds"
        assert trajectory_id < len(self._offsets), "trajectory_id out of bounds"

        payload = self._load_payload_at_offset(self._offsets[trajectory_id])
        return _parse_direct_training_trajectory_payload(
            trajectory_id=trajectory_id, payload=payload
        )


def iter_tokenized_samples(msgpack_path: str) -> Iterator[TrainingSampleTokenized]:
    for payload in _iter_msgpack_payloads(msgpack_path):
        yield _parse_tokenized_payload(payload)


def load_tokenized_samples(msgpack_path: str) -> list[TrainingSampleTokenized]:
    return list(iter_tokenized_samples(msgpack_path))


def iter_training_trajectories(msgpack_path: str) -> Iterator[TrainingSampleTokenized]:
    for trajectory_id, payload in enumerate(_iter_msgpack_payloads(msgpack_path)):
        yield _parse_direct_training_trajectory_payload(trajectory_id, payload)


def load_training_trajectories(msgpack_path: str) -> list[TrainingSampleTokenized]:
    return list(iter_training_trajectories(msgpack_path))
