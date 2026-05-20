import tempfile
import unittest

from research_utility import SqliteStore

from src_py.train.data_sqlite import (
    load_tokenized_samples,
    iter_training_batches,
)


def _write_entries(db_path: str, entries: list[tuple[str, object]]) -> None:
    store = SqliteStore[str, object](db_path)
    try:
        store.clear()
        for entry_id, payload in entries:
            store.upsert(entry_id, payload)
    finally:
        store.close()


class TestDataSqlite(unittest.TestCase):
    def test_load_tokenized_samples_parses_and_orders_by_id(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as temp_db:
            payload_b = {
                "id": {"question_id": 2, "node_id": 5},
                "input_ids": [11, 12, 13],
                "labels": [-100, 12, 13],
                "reconstructed": "sample b",
                "input_length": 3,
                "advantage": -0.25,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            payload_a = {
                "id": {"question_id": 1, "node_id": 3},
                "input_ids": [7, 8],
                "labels": [-100, 8],
                "reconstructed": "sample a",
                "input_length": 2,
                "advantage": 1.5,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }

            _write_entries(
                temp_db.name,
                [
                    ("q2_n5", payload_b),
                    ("q1_n3", payload_a),
                ],
            )

            samples = load_tokenized_samples(temp_db.name)
            self.assertEqual(2, len(samples))

            self.assertEqual(1, samples[0].id.question_id)
            self.assertEqual(3, samples[0].id.node_id)
            self.assertEqual([7, 8], samples[0].input_ids)
            self.assertEqual([-100, 8], samples[0].labels)
            self.assertEqual(1.5, samples[0].advantage)

            self.assertEqual(2, samples[1].id.question_id)
            self.assertEqual(5, samples[1].id.node_id)

    def test_iter_training_batches_orders_numerically(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as temp_db:
            entries: list[tuple[str, object]] = []
            for batch_id in ["10", "2", "0", "1"]:
                index = int(batch_id)
                payload = {
                    "ids": [{"question_id": index, "node_id": index + 1}],
                    "max_advantage": float(index),
                    "min_advantage": float(index),
                    "max_length": index + 100,
                    "min_length": index + 100,
                    "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
                }
                entries.append((batch_id, payload))

            _write_entries(temp_db.name, entries)

            with self.assertRaises(AssertionError):
                # The iterator reads in numeric order: 0, 1, 2, 10.
                # It should fail fast because contiguous IDs are required.
                list(iter_training_batches(temp_db.name))

    def test_iter_training_batches_rejects_non_numeric_batch_id(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as temp_db:
            valid_payload = {
                "ids": [{"question_id": 0, "node_id": 0}],
                "max_advantage": 0.0,
                "min_advantage": 0.0,
                "max_length": 8,
                "min_length": 8,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            _write_entries(
                temp_db.name,
                [
                    ("0", valid_payload),
                    ("bad_id", valid_payload),
                ],
            )

            with self.assertRaises(AssertionError):
                list(iter_training_batches(temp_db.name))


if __name__ == "__main__":
    unittest.main()
