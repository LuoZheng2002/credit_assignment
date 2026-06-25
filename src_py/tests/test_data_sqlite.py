import tempfile
import unittest
from typing import cast

import msgpack

from src_py.train.data_msgpack import (
    LazyTrainingTrajectoryStore,
    iter_training_trajectories,
)


def _write_entries(msgpack_path: str, entries: list[object]) -> None:
    with open(msgpack_path, "wb") as file:
        for payload in entries:
            file.write(
                cast(bytes, msgpack.packb(cast(object, payload), use_bin_type=True))
            )


class TestDataSqlite(unittest.TestCase):
    def test_iter_training_trajectories_preserves_file_order(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".msgpack") as temp_file:
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
                "advantages": [0.0, -0.25, -0.75],
                "average_absolute_segment_advantage": 0.5,
            }
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
                "advantages": [0.0, 1.5],
                "average_absolute_segment_advantage": 0.8,
            }

            _write_entries(temp_file.name, [payload_zero, payload_one])

            samples = list(iter_training_trajectories(temp_file.name))
            self.assertEqual(2, len(samples))

            self.assertEqual(2, samples[0].id.question_id)
            self.assertEqual(0, samples[0].id.node_id)
            self.assertEqual([11, 12, 13], samples[0].input_ids)
            self.assertEqual([-100, 12, 13], samples[0].labels)
            self.assertEqual([0.0, -0.25, -0.75], samples[0].token_advantages)

            self.assertEqual(1, samples[1].id.question_id)
            self.assertEqual(1, samples[1].id.node_id)
            self.assertEqual([0.0, 1.5], samples[1].token_advantages)

    def test_lazy_training_trajectory_store_reads_by_index(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".msgpack") as temp_file:
            payload_zero = {
                "question": {
                    "flat_id": 0,
                    "dataset_name": "deepmath",
                    "question_id": 0,
                    "question": "q0",
                    "correct_answer": "a0",
                },
                "input_ids": [1, 2],
                "labels": [-100, 2],
                "advantages": [0.0, 0.5],
                "average_absolute_segment_advantage": 0.5,
            }
            payload_one = {
                "question": {
                    "flat_id": 1,
                    "dataset_name": "deepmath",
                    "question_id": 1,
                    "question": "q1",
                    "correct_answer": "a1",
                },
                "input_ids": [3, 4, 5],
                "labels": [-100, 4, 5],
                "advantages": [0.0, 0.25, 0.25],
                "average_absolute_segment_advantage": 0.25,
            }
            _write_entries(temp_file.name, [payload_zero, payload_one])

            store = LazyTrainingTrajectoryStore(temp_file.name)
            try:
                self.assertEqual(2, store.sample_count)
                sample = store.get_sample(1)
                self.assertEqual(1, sample.id.question_id)
                self.assertEqual([3, 4, 5], sample.input_ids)
            finally:
                store.close()

    def test_iter_training_trajectories_rejects_missing_supervised_tokens(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".msgpack") as temp_file:
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
            _write_entries(temp_file.name, [payload])

            with self.assertRaises(AssertionError):
                list(iter_training_trajectories(temp_file.name))


if __name__ == "__main__":
    unittest.main()
