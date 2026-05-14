from __future__ import annotations

from dataclasses import dataclass

from .data_sqlite import (
    QuestionNodeId,
    TrainingSampleTokenized,
    iter_training_batches,
    iter_tokenized_samples,
)


@dataclass(frozen=True)
class ResolvedTrainingBatch:
    batch_index: int
    ids: list[QuestionNodeId]
    samples: list[TrainingSampleTokenized]
    model_official_name: str


def _question_node_key(sample_id: QuestionNodeId) -> tuple[int, int]:
    return (sample_id.question_id, sample_id.node_id)


def load_resolved_training_batches(
    tokenized_sqlite_path: str,
    batch_sqlite_path: str,
) -> list[ResolvedTrainingBatch]:
    tokenized_by_id: dict[tuple[int, int], TrainingSampleTokenized] = {}
    for sample in iter_tokenized_samples(tokenized_sqlite_path):
        key = _question_node_key(sample.id)
        assert key not in tokenized_by_id, f"duplicate sample id detected: {sample.id}"
        tokenized_by_id[key] = sample

    assert len(tokenized_by_id) > 0, "tokenized sample database must be non-empty"

    resolved_batches: list[ResolvedTrainingBatch] = []
    seen_ids: set[tuple[int, int]] = set()
    batch_model_names_seen: set[str] = set()

    for batch in iter_training_batches(batch_sqlite_path):
        batch_model_names_seen.add(batch.model_official_name)
        resolved_samples: list[TrainingSampleTokenized] = []
        for sample_id in batch.ids:
            key = _question_node_key(sample_id)
            assert key in tokenized_by_id, f"missing tokenized sample for id: {sample_id}"
            resolved_sample = tokenized_by_id[key]
            assert (
                resolved_sample.input_length == len(resolved_sample.input_ids)
            ), "input_length must equal len(input_ids)"
            assert (
                resolved_sample.model_official_name == batch.model_official_name
            ), "tokenized sample model_official_name must match batch model_official_name"
            resolved_samples.append(resolved_sample)
            seen_ids.add(key)

        assert len(resolved_samples) > 0, "resolved batch cannot be empty"
        resolved_batches.append(
            ResolvedTrainingBatch(
                batch_index=batch.batch_index,
                ids=batch.ids,
                samples=resolved_samples,
                model_official_name=batch.model_official_name,
            )
        )

    assert len(resolved_batches) > 0, "batch database must be non-empty"
    assert len(seen_ids) > 0, "at least one sample id must appear in batches"
    assert (
        len(batch_model_names_seen) == 1
    ), "all training batches must have the same model_official_name"

    return resolved_batches
