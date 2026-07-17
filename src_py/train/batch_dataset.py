from __future__ import annotations

from dataclasses import dataclass

from .data_msgpack import (
    LazyTrainingTrajectoryStore,
    QuestionNodeId,
    TrainingSampleTokenized,
    iter_training_trajectories,
)


def _memtrace(message: str) -> None:
    print(f"[memtrace] {message}", flush=True)


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
    training_trajectory_path: str,
    batch_size: int,
    model_official_name: str,
    first_n_training_samples: int,
    training_trajectory_len_cutoff: int,
) -> list[ResolvedTrainingBatch]:
    assert batch_size > 0, "batch_size must be positive"
    assert len(model_official_name.strip()) > 0, "model_official_name cannot be empty"
    assert first_n_training_samples >= 0, (
        "first_n_training_samples must be non-negative"
    )
    assert training_trajectory_len_cutoff >= 2, (
        "training_trajectory_len_cutoff must be at least 2"
    )

    trajectories: list[TrainingSampleTokenized] = []
    tokenized_by_id: dict[tuple[int, int], TrainingSampleTokenized] = {}
    for sample in iter_training_trajectories(training_trajectory_path):
        key = _question_node_key(sample.id)
        assert key not in tokenized_by_id, f"duplicate sample id detected: {sample.id}"
        if sample.input_length > training_trajectory_len_cutoff:
            continue
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
        training_trajectory_path: str,
        training_trajectory_len_cutoff: int,
        model_official_name: str,
        first_n_training_samples: int,
    ):
        assert len(model_official_name.strip()) > 0, (
            "model_official_name cannot be empty"
        )
        assert training_trajectory_len_cutoff >= 2, (
            "training_trajectory_len_cutoff must be at least 2"
        )
        _memtrace(
            "lazy_batch_loader_init_begin "
            f"path={training_trajectory_path} training_trajectory_len_cutoff={training_trajectory_len_cutoff}"
        )
        self._store = LazyTrainingTrajectoryStore(
            msgpack_path=training_trajectory_path,
            first_n_training_samples=first_n_training_samples,
            training_trajectory_len_cutoff=training_trajectory_len_cutoff,
        )
        self.sample_count = self._store.sample_count
        self._model_official_name = model_official_name
        _memtrace(
            "lazy_batch_loader_init_end "
            f"sample_count={self.sample_count} model_official_name={model_official_name}"
        )

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
        emit_debug = sample_index == 0 or batch_index < 3
        if emit_debug:
            _memtrace(
                "resolve_batch_begin "
                f"sample_index={sample_index} batch_size={batch_size} batch_index={batch_index}"
            )

        end_sample_index = min(self.sample_count, sample_index + batch_size)
        samples: list[TrainingSampleTokenized] = []
        ids: list[QuestionNodeId] = []
        for trajectory_id in range(sample_index, end_sample_index):
            sample = self._store.get_sample(trajectory_id)
            samples.append(sample)
            ids.append(sample.id)

        assert len(samples) > 0, "resolved batch cannot be empty"
        if emit_debug:
            lengths = [sample.input_length for sample in samples]
            _memtrace(
                "resolve_batch_end "
                f"batch_index={batch_index} resolved_samples={len(samples)} "
                f"min_input_length={min(lengths)} max_input_length={max(lengths)} "
                f"next_sample_index={end_sample_index}"
            )

        return LazyBatchWindow(
            resolved_batch=ResolvedTrainingBatch(
                batch_index=batch_index,
                ids=ids,
                samples=samples,
                model_official_name=self._model_official_name,
            ),
            next_sample_index=end_sample_index,
        )
