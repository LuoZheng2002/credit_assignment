from __future__ import annotations

from dataclasses import dataclass

from .data_msgpack import (
    LazyTrainingTrajectoryStore,
    QuestionNodeId,
    TrainingSampleTokenized,
    iter_training_trajectories,
)


@dataclass(frozen=True)
class ResolvedTrainingBatch:
    batch_index: int
    ids: list[QuestionNodeId]
    samples: list[TrainingSampleTokenized]
    model_official_name: str


@dataclass(frozen=True)
class LazyBatchWindow:
    resolved_batch: ResolvedTrainingBatch
    next_sample_index: int


def _question_node_key(sample_id: QuestionNodeId) -> tuple[int, int]:
    return (sample_id.question_id, sample_id.node_id)


def load_resolved_training_batches(
    training_trajectory_sqlite_path: str,
    batch_size: int,
    model_official_name: str,
    first_n_training_samples: int,
) -> list[ResolvedTrainingBatch]:
    assert batch_size > 0, "batch_size must be positive"
    assert len(model_official_name.strip()) > 0, "model_official_name cannot be empty"
    assert first_n_training_samples >= 0, (
        "first_n_training_samples must be non-negative"
    )

    trajectories: list[TrainingSampleTokenized] = []
    tokenized_by_id: dict[tuple[int, int], TrainingSampleTokenized] = {}
    previous_input_length = -1
    for sample in iter_training_trajectories(training_trajectory_sqlite_path):
        key = _question_node_key(sample.id)
        assert key not in tokenized_by_id, f"duplicate sample id detected: {sample.id}"
        assert (
            sample.input_length <= previous_input_length or previous_input_length == -1
        ), (
            "training trajectories must be sorted by input_ids length in descending order"
        )
        previous_input_length = sample.input_length
        tokenized_by_id[key] = sample
        trajectories.append(sample)

        if (
            first_n_training_samples > 0
            and len(trajectories) == first_n_training_samples
        ):
            break

    assert len(trajectories) > 0, "training trajectory database must be non-empty"

    resolved_batches: list[ResolvedTrainingBatch] = []
    seen_ids: set[tuple[int, int]] = set()

    for batch_start in range(0, len(trajectories), batch_size):
        batch_index = len(resolved_batches)
        resolved_samples = trajectories[batch_start : batch_start + batch_size]
        ids: list[QuestionNodeId] = []
        for sample in resolved_samples:
            key = _question_node_key(sample.id)
            assert key in tokenized_by_id, (
                f"missing tokenized sample for id: {sample.id}"
            )
            ids.append(sample.id)
            seen_ids.add(key)

        assert len(resolved_samples) > 0, "resolved batch cannot be empty"
        resolved_batches.append(
            ResolvedTrainingBatch(
                batch_index=batch_index,
                ids=ids,
                samples=resolved_samples,
                model_official_name=model_official_name,
            )
        )

    assert len(resolved_batches) > 0, "batch database must be non-empty"
    assert len(seen_ids) > 0, "at least one sample id must appear in batches"

    return resolved_batches


class LazyResolvedBatchLoader:
    def __init__(
        self,
        training_trajectory_sqlite_path: str,
        model_official_name: str,
        first_n_training_samples: int,
    ):
        assert len(model_official_name.strip()) > 0, (
            "model_official_name cannot be empty"
        )
        self._store = LazyTrainingTrajectoryStore(
            msgpack_path=training_trajectory_sqlite_path,
            first_n_training_samples=first_n_training_samples,
        )
        self.sample_count = self._store.sample_count
        self._model_official_name = model_official_name

    def close(self) -> None:
        self._store.close()

    def get_sample(self, sample_index: int) -> TrainingSampleTokenized:
        assert sample_index >= 0, "sample_index must be non-negative"
        assert sample_index < self.sample_count, "sample_index out of range"
        return self._store.get_sample(sample_index)

    def resolve_batch(
        self, sample_index: int, batch_size: int, batch_index: int
    ) -> LazyBatchWindow:
        assert sample_index >= 0, "sample_index must be non-negative"
        assert sample_index < self.sample_count, "sample_index out of range"
        assert batch_size > 0, "batch_size must be positive"
        assert batch_index >= 0, "batch_index must be non-negative"

        end_sample_index = min(self.sample_count, sample_index + batch_size)
        samples: list[TrainingSampleTokenized] = []
        ids: list[QuestionNodeId] = []
        previous_length = -1
        for trajectory_id in range(sample_index, end_sample_index):
            sample = self._store.get_sample(trajectory_id)
            assert sample.input_length <= previous_length or previous_length == -1, (
                "training trajectories must be sorted by input_ids length in descending order"
            )
            previous_length = sample.input_length
            samples.append(sample)
            ids.append(sample.id)

        assert len(samples) > 0, "resolved batch cannot be empty"

        return LazyBatchWindow(
            resolved_batch=ResolvedTrainingBatch(
                batch_index=batch_index,
                ids=ids,
                samples=samples,
                model_official_name=self._model_official_name,
            ),
            next_sample_index=end_sample_index,
        )
