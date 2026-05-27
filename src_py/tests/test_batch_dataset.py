import tempfile
import unittest
from typing import Any

from research_utility import SqliteStore

from src_py.train.batch_dataset import load_resolved_training_batches


def _write_entries(db_path: str, entries: list[tuple[Any, object]]) -> None:
    store = SqliteStore[Any, object](db_path)
    try:
        store.clear()
        for entry_id, payload in entries:
            store.upsert(entry_id, payload)
    finally:
        store.close()


class TestBatchDataset(unittest.TestCase):
    def test_load_resolved_training_batches_chunks_ascending_trajectories(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as trajectory_db:
            trajectory_0 = {
                "question": {
                    "flat_id": 100,
                    "dataset_name": "deepmath",
                    "question_id": 1,
                    "question": "q1",
                    "correct_answer": "a1",
                },
                "input_ids": [11, 12],
                "labels": [-100, 12],
                "advantages": [0.1, 0.3],
                "average_absolute_segment_advantage": 0.2,
            }
            trajectory_1 = {
                "question": {
                    "flat_id": 101,
                    "dataset_name": "deepmath",
                    "question_id": 2,
                    "question": "q2",
                    "correct_answer": "a2",
                },
                "input_ids": [21, 22, 23],
                "labels": [-100, 22, 23],
                "advantages": [-0.7, -0.7, -0.7],
                "average_absolute_segment_advantage": 0.7,
            }
            trajectory_2 = {
                "question": {
                    "flat_id": 102,
                    "dataset_name": "deepmath",
                    "question_id": 3,
                    "question": "q3",
                    "correct_answer": "a3",
                },
                "input_ids": [31, 32, 33, 34],
                "labels": [-100, 32, 33, 34],
                "advantages": [0.2, 0.2, 0.2, 0.2],
                "average_absolute_segment_advantage": 0.2,
            }

            _write_entries(
                trajectory_db.name,
                [
                    (0, trajectory_0),
                    (1, trajectory_1),
                    (2, trajectory_2),
                ],
            )

            batches = load_resolved_training_batches(
                training_trajectory_sqlite_path=trajectory_db.name,
                batch_size=2,
                model_official_name="Qwen/Qwen2.5-7B-Instruct",
                first_n_training_samples=0,
            )

            self.assertEqual(2, len(batches))
            self.assertEqual(0, batches[0].batch_index)
            self.assertEqual(2, len(batches[0].samples))
            self.assertEqual([2, 3], [sample.input_length for sample in batches[0].samples])
            self.assertEqual(1, batches[1].batch_index)
            self.assertEqual(1, len(batches[1].samples))
            self.assertEqual(4, batches[1].samples[0].input_length)

    def test_load_resolved_training_batches_requires_ascending_lengths(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as trajectory_db:
            trajectory_0 = {
                "question": {
                    "flat_id": 100,
                    "dataset_name": "deepmath",
                    "question_id": 1,
                    "question": "q1",
                    "correct_answer": "a1",
                },
                "input_ids": [11, 12, 13],
                "labels": [-100, 12, 13],
                "advantages": [0.1, 0.1, 0.1],
                "average_absolute_segment_advantage": 0.1,
            }
            trajectory_1 = {
                "question": {
                    "flat_id": 101,
                    "dataset_name": "deepmath",
                    "question_id": 2,
                    "question": "q2",
                    "correct_answer": "a2",
                },
                "input_ids": [21, 22],
                "labels": [-100, 22],
                "advantages": [0.2, 0.2],
                "average_absolute_segment_advantage": 0.2,
            }
            _write_entries(trajectory_db.name, [(0, trajectory_0), (1, trajectory_1)])

            with self.assertRaises(AssertionError):
                load_resolved_training_batches(
                    training_trajectory_sqlite_path=trajectory_db.name,
                    batch_size=2,
                    model_official_name="Qwen/Qwen2.5-7B-Instruct",
                    first_n_training_samples=0,
                )

    def test_load_resolved_training_batches_honors_first_n_training_samples(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as trajectory_db:
            trajectory_entries: list[tuple[int, object]] = []
            for index in range(4):
                trajectory_entries.append(
                    (
                        index,
                        {
                            "question": {
                                "flat_id": 100 + index,
                                "dataset_name": "deepmath",
                                "question_id": index,
                                "question": f"q{index}",
                                "correct_answer": f"a{index}",
                            },
                            "input_ids": [10 + index, 20 + index],
                            "labels": [-100, 20 + index],
                            "advantages": [0.1, 0.2],
                            "average_absolute_segment_advantage": 0.15,
                        },
                    )
                )
            _write_entries(trajectory_db.name, trajectory_entries)

            batches = load_resolved_training_batches(
                training_trajectory_sqlite_path=trajectory_db.name,
                batch_size=2,
                model_official_name="Qwen/Qwen2.5-7B-Instruct",
                first_n_training_samples=3,
            )

            self.assertEqual(2, len(batches))
            self.assertEqual(2, len(batches[0].samples))
            self.assertEqual(1, len(batches[1].samples))


if __name__ == "__main__":
    unittest.main()
