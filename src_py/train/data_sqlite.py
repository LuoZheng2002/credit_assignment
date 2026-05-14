from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import sqlite3
from typing import Iterator


SQLITE_TABLE_NAME = "store_entries"


@dataclass(frozen=True)
class QuestionNodeId:
    question_id: int
    node_id: int


@dataclass(frozen=True)
class TrainingSampleTokenized:
    id: QuestionNodeId
    input_ids: list[int]
    labels: list[int]
    reconstructed: str
    input_length: int
    advantage: float
    model_official_name: str


@dataclass(frozen=True)
class TrainingBatch:
    batch_index: int
    ids: list[QuestionNodeId]
    max_advantage: float
    min_advantage: float
    max_length: int
    min_length: int
    model_official_name: str


def _open_connection(sqlite_path: str) -> sqlite3.Connection:
    assert os.path.isfile(sqlite_path), f"sqlite file not found: {sqlite_path}"
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_question_node_id(value: object) -> QuestionNodeId:
    assert isinstance(value, dict), "QuestionNodeId must be a JSON object"
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
    assert isinstance(value, list), f"{field_name} must be a JSON array"
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


def _parse_tokenized_payload(payload_json: str) -> TrainingSampleTokenized:
    payload_obj = json.loads(payload_json)
    assert isinstance(payload_obj, dict), "tokenized payload must be a JSON object"
    assert "id" in payload_obj, "tokenized payload must contain id"
    assert "input_ids" in payload_obj, "tokenized payload must contain input_ids"
    assert "labels" in payload_obj, "tokenized payload must contain labels"
    assert "reconstructed" in payload_obj, "tokenized payload must contain reconstructed"
    assert "input_length" in payload_obj, "tokenized payload must contain input_length"
    assert "advantage" in payload_obj, "tokenized payload must contain advantage"
    assert (
        "model_official_name" in payload_obj
    ), "tokenized payload must contain model_official_name"

    sample_id = _parse_question_node_id(payload_obj["id"])
    input_ids = _parse_int_list(payload_obj["input_ids"], "input_ids")
    labels = _parse_int_list(payload_obj["labels"], "labels")
    reconstructed_obj = payload_obj["reconstructed"]
    assert isinstance(reconstructed_obj, str), "reconstructed must be string"
    input_length = _parse_positive_int(payload_obj["input_length"], "input_length")
    advantage = _parse_finite_float(payload_obj["advantage"], "advantage")
    model_official_name_obj = payload_obj["model_official_name"]
    assert isinstance(model_official_name_obj, str), "model_official_name must be string"
    assert len(model_official_name_obj) > 0, "model_official_name cannot be empty"

    assert len(input_ids) > 0, "input_ids cannot be empty"
    assert len(labels) == len(input_ids), "labels and input_ids lengths must match"
    assert input_length == len(input_ids), "input_length must equal len(input_ids)"

    return TrainingSampleTokenized(
        id=sample_id,
        input_ids=input_ids,
        labels=labels,
        reconstructed=reconstructed_obj,
        input_length=input_length,
        advantage=advantage,
        model_official_name=model_official_name_obj,
    )


def _parse_batch_payload(batch_index: int, payload_json: str) -> TrainingBatch:
    payload_obj = json.loads(payload_json)
    assert isinstance(payload_obj, dict), "batch payload must be a JSON object"
    assert "ids" in payload_obj, "batch payload must contain ids"
    assert "max_advantage" in payload_obj, "batch payload must contain max_advantage"
    assert "min_advantage" in payload_obj, "batch payload must contain min_advantage"
    assert "max_length" in payload_obj, "batch payload must contain max_length"
    assert "min_length" in payload_obj, "batch payload must contain min_length"
    assert "model_official_name" in payload_obj, "batch payload must contain model_official_name"

    ids_obj = payload_obj["ids"]
    assert isinstance(ids_obj, list), "ids must be a JSON array"
    ids: list[QuestionNodeId] = []
    for id_obj in ids_obj:
        ids.append(_parse_question_node_id(id_obj))

    assert len(ids) > 0, "batch ids cannot be empty"

    max_advantage = _parse_finite_float(payload_obj["max_advantage"], "max_advantage")
    min_advantage = _parse_finite_float(payload_obj["min_advantage"], "min_advantage")
    max_length = _parse_positive_int(payload_obj["max_length"], "max_length")
    min_length = _parse_positive_int(payload_obj["min_length"], "min_length")
    model_official_name_obj = payload_obj["model_official_name"]
    assert isinstance(model_official_name_obj, str), "model_official_name must be string"
    assert len(model_official_name_obj) > 0, "model_official_name cannot be empty"

    assert max_length >= min_length, "max_length must be >= min_length"
    assert max_advantage >= min_advantage, "max_advantage must be >= min_advantage"

    return TrainingBatch(
        batch_index=batch_index,
        ids=ids,
        max_advantage=max_advantage,
        min_advantage=min_advantage,
        max_length=max_length,
        min_length=min_length,
        model_official_name=model_official_name_obj,
    )


def iter_tokenized_samples(sqlite_path: str) -> Iterator[TrainingSampleTokenized]:
    connection = _open_connection(sqlite_path)
    try:
        cursor = connection.execute(
            f"SELECT payload_json FROM {SQLITE_TABLE_NAME} ORDER BY id ASC"
        )
        for row in cursor:
            payload_json_obj = row["payload_json"]
            assert isinstance(payload_json_obj, str), "payload_json must be string"
            yield _parse_tokenized_payload(payload_json_obj)
    finally:
        connection.close()


def load_tokenized_samples(sqlite_path: str) -> list[TrainingSampleTokenized]:
    return list(iter_tokenized_samples(sqlite_path))


def iter_training_batches(sqlite_path: str) -> Iterator[TrainingBatch]:
    connection = _open_connection(sqlite_path)
    try:
        cursor = connection.execute(
            f"SELECT id, payload_json FROM {SQLITE_TABLE_NAME} ORDER BY CAST(id AS INTEGER) ASC"
        )
        expected_batch_index = 0
        for row in cursor:
            id_obj = row["id"]
            payload_json_obj = row["payload_json"]
            assert isinstance(id_obj, str), "batch id must be string"
            assert isinstance(payload_json_obj, str), "payload_json must be string"
            assert id_obj.isdigit(), "batch id must be numeric text"

            batch_index = int(id_obj)
            assert (
                batch_index == expected_batch_index
            ), f"batch index must be contiguous: expected {expected_batch_index}, got {batch_index}"

            yield _parse_batch_payload(batch_index=batch_index, payload_json=payload_json_obj)
            expected_batch_index += 1
    finally:
        connection.close()


def load_training_batches(sqlite_path: str) -> list[TrainingBatch]:
    return list(iter_training_batches(sqlite_path))
