import tempfile
import unittest

from research_utility import SqliteStore

from src_py.train.batch_dataset import load_resolved_training_batches


def _write_entries(db_path: str, entries: list[tuple[str, object]]) -> None:
    store = SqliteStore[str, object](db_path)
    try:
        store.clear()
        for entry_id, payload in entries:
            store.upsert(entry_id, payload)
    finally:
        store.close()


class TestBatchDataset(unittest.TestCase):
    def test_load_resolved_training_batches_respects_batch_order(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tokenized_db, tempfile.NamedTemporaryFile(
            suffix=".sqlite"
        ) as batch_db:
            sample_q1_n1 = {
                "id": {"question_id": 1, "node_id": 1},
                "input_ids": [11, 12],
                "labels": [-100, 12],
                "reconstructed": "s1",
                "input_length": 2,
                "advantage": 0.3,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            sample_q2_n2 = {
                "id": {"question_id": 2, "node_id": 2},
                "input_ids": [21, 22, 23],
                "labels": [-100, 22, 23],
                "reconstructed": "s2",
                "input_length": 3,
                "advantage": -0.7,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }

            _write_entries(
                tokenized_db.name,
                [
                    ("q2_n2", sample_q2_n2),
                    ("q1_n1", sample_q1_n1),
                ],
            )

            batch_zero = {
                "ids": [{"question_id": 2, "node_id": 2}],
                "max_advantage": -0.7,
                "min_advantage": -0.7,
                "max_length": 3,
                "min_length": 3,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            batch_one = {
                "ids": [{"question_id": 1, "node_id": 1}],
                "max_advantage": 0.3,
                "min_advantage": 0.3,
                "max_length": 2,
                "min_length": 2,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }

            _write_entries(
                batch_db.name,
                [
                    ("0", batch_zero),
                    ("1", batch_one),
                ],
            )

            batches = load_resolved_training_batches(tokenized_db.name, batch_db.name)

            self.assertEqual(2, len(batches))
            self.assertEqual(0, batches[0].batch_index)
            self.assertEqual(2, batches[0].samples[0].id.question_id)
            self.assertEqual(1, batches[1].batch_index)
            self.assertEqual(1, batches[1].samples[0].id.question_id)

    def test_load_resolved_training_batches_fails_on_missing_sample(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tokenized_db, tempfile.NamedTemporaryFile(
            suffix=".sqlite"
        ) as batch_db:
            tokenized_sample = {
                "id": {"question_id": 1, "node_id": 1},
                "input_ids": [101],
                "labels": [101],
                "reconstructed": "single",
                "input_length": 1,
                "advantage": 0.0,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            _write_entries(tokenized_db.name, [("q1_n1", tokenized_sample)])

            missing_batch = {
                "ids": [{"question_id": 9, "node_id": 9}],
                "max_advantage": 1.0,
                "min_advantage": 1.0,
                "max_length": 1,
                "min_length": 1,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            _write_entries(batch_db.name, [("0", missing_batch)])

            with self.assertRaises(AssertionError):
                load_resolved_training_batches(tokenized_db.name, batch_db.name)

    def test_load_resolved_training_batches_fails_on_model_name_mismatch(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tokenized_db, tempfile.NamedTemporaryFile(
            suffix=".sqlite"
        ) as batch_db:
            tokenized_sample = {
                "id": {"question_id": 1, "node_id": 1},
                "input_ids": [9],
                "labels": [9],
                "reconstructed": "single",
                "input_length": 1,
                "advantage": 0.2,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            _write_entries(tokenized_db.name, [("q1_n1", tokenized_sample)])

            mismatched_batch = {
                "ids": [{"question_id": 1, "node_id": 1}],
                "max_advantage": 0.2,
                "min_advantage": 0.2,
                "max_length": 1,
                "min_length": 1,
                "model_official_name": "Qwen/Qwen3-4B",
            }
            _write_entries(batch_db.name, [("0", mismatched_batch)])

            with self.assertRaises(AssertionError):
                load_resolved_training_batches(tokenized_db.name, batch_db.name)


if __name__ == "__main__":
    unittest.main()
