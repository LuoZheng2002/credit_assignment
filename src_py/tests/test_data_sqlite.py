import tempfile
import unittest
from typing import Any

from research_utility import SqliteStore

from src_py.train.data_sqlite import (
    iter_training_trajectories,
)


def _write_entries(db_path: str, entries: list[tuple[Any, object]]) -> None:
    store = SqliteStore[Any, object](db_path)
    try:
        store.clear()
        for entry_id, payload in entries:
            store.upsert(entry_id, payload)
    finally:
        store.close()


class TestDataSqlite(unittest.TestCase):
    def test_iter_training_trajectories_parses_and_orders_by_index(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as temp_db:
            payload_one = {
                "question": {
                    "flat_id": 1,
                    "dataset_name": "deepmath",
                    "question_id": 10,
                    "question": "q1",
                    "correct_answer": "a1",
                },
                "input_ids": [7, 8],
                "labels": [-100, 8],
                "advantages": [0.1, 1.5],
                "average_absolute_segment_advantage": 0.8,
            }
            payload_zero = {
                "question": {
                    "flat_id": 2,
                    "dataset_name": "deepmath",
                    "question_id": 20,
                    "question": "q2",
                    "correct_answer": "a2",
                },
                "input_ids": [11, 12, 13],
                "labels": [-100, 12, 13],
                "advantages": [0.2, -0.25, -0.75],
                "average_absolute_segment_advantage": 0.5,
            }

            _write_entries(
                temp_db.name,
                [
                    (1, payload_one),
                    (0, payload_zero),
                ],
            )

            samples = list(iter_training_trajectories(temp_db.name))
            self.assertEqual(2, len(samples))

            self.assertEqual(2, samples[0].id.question_id)
            self.assertEqual(0, samples[0].id.node_id)
            self.assertEqual([11, 12, 13], samples[0].input_ids)
            self.assertEqual([-100, 12, 13], samples[0].labels)
            self.assertAlmostEqual(-0.5, samples[0].advantage, places=6)

            self.assertEqual(1, samples[1].id.question_id)
            self.assertEqual(1, samples[1].id.node_id)

    def test_iter_training_trajectories_requires_contiguous_indices(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as temp_db:
            payload = {
                "question": {
                    "flat_id": 0,
                    "dataset_name": "deepmath",
                    "question_id": 0,
                    "question": "q",
                    "correct_answer": "a",
                },
                "input_ids": [1, 2],
                "labels": [-100, 2],
                "advantages": [0.0, 0.5],
                "average_absolute_segment_advantage": 0.5,
            }
            _write_entries(
                temp_db.name,
                [
                    (0, payload),
                    (2, payload),
                ],
            )

            with self.assertRaises(AssertionError):
                list(iter_training_trajectories(temp_db.name))

    def test_iter_training_trajectories_rejects_missing_supervised_tokens(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as temp_db:
            payload = {
                "question": {
                    "flat_id": 0,
                    "dataset_name": "deepmath",
                    "question_id": 0,
                    "question": "q",
                    "correct_answer": "a",
                },
                "input_ids": [1, 2],
                "labels": [-100, -100],
                "advantages": [0.0, 0.5],
                "average_absolute_segment_advantage": 0.5,
            }
            _write_entries(temp_db.name, [(0, payload)])

            with self.assertRaises(AssertionError):
                list(iter_training_trajectories(temp_db.name))


if __name__ == "__main__":
    unittest.main()
