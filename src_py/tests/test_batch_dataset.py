import json
import sqlite3
import tempfile
import unittest

from src_py.train.batch_dataset import load_resolved_training_batches


def _create_store_entries_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE store_entries (
            id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.commit()


class TestBatchDataset(unittest.TestCase):
    def test_load_resolved_training_batches_respects_batch_order(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tokenized_db, tempfile.NamedTemporaryFile(
            suffix=".sqlite"
        ) as batch_db:
            tokenized_connection = sqlite3.connect(tokenized_db.name)
            _create_store_entries_table(tokenized_connection)

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

            tokenized_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("q2_n2", json.dumps(sample_q2_n2)),
            )
            tokenized_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("q1_n1", json.dumps(sample_q1_n1)),
            )
            tokenized_connection.commit()
            tokenized_connection.close()

            batch_connection = sqlite3.connect(batch_db.name)
            _create_store_entries_table(batch_connection)

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

            batch_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("0", json.dumps(batch_zero)),
            )
            batch_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("1", json.dumps(batch_one)),
            )
            batch_connection.commit()
            batch_connection.close()

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
            tokenized_connection = sqlite3.connect(tokenized_db.name)
            _create_store_entries_table(tokenized_connection)
            tokenized_sample = {
                "id": {"question_id": 1, "node_id": 1},
                "input_ids": [101],
                "labels": [101],
                "reconstructed": "single",
                "input_length": 1,
                "advantage": 0.0,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            tokenized_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("q1_n1", json.dumps(tokenized_sample)),
            )
            tokenized_connection.commit()
            tokenized_connection.close()

            batch_connection = sqlite3.connect(batch_db.name)
            _create_store_entries_table(batch_connection)
            missing_batch = {
                "ids": [{"question_id": 9, "node_id": 9}],
                "max_advantage": 1.0,
                "min_advantage": 1.0,
                "max_length": 1,
                "min_length": 1,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            batch_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("0", json.dumps(missing_batch)),
            )
            batch_connection.commit()
            batch_connection.close()

            with self.assertRaises(AssertionError):
                load_resolved_training_batches(tokenized_db.name, batch_db.name)

    def test_load_resolved_training_batches_fails_on_model_name_mismatch(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tokenized_db, tempfile.NamedTemporaryFile(
            suffix=".sqlite"
        ) as batch_db:
            tokenized_connection = sqlite3.connect(tokenized_db.name)
            _create_store_entries_table(tokenized_connection)
            tokenized_sample = {
                "id": {"question_id": 1, "node_id": 1},
                "input_ids": [9],
                "labels": [9],
                "reconstructed": "single",
                "input_length": 1,
                "advantage": 0.2,
                "model_official_name": "Qwen/Qwen2.5-7B-Instruct",
            }
            tokenized_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("q1_n1", json.dumps(tokenized_sample)),
            )
            tokenized_connection.commit()
            tokenized_connection.close()

            batch_connection = sqlite3.connect(batch_db.name)
            _create_store_entries_table(batch_connection)
            mismatched_batch = {
                "ids": [{"question_id": 1, "node_id": 1}],
                "max_advantage": 0.2,
                "min_advantage": 0.2,
                "max_length": 1,
                "min_length": 1,
                "model_official_name": "Qwen/Qwen3-4B",
            }
            batch_connection.execute(
                "INSERT INTO store_entries (id, payload_json) VALUES (?, ?)",
                ("0", json.dumps(mismatched_batch)),
            )
            batch_connection.commit()
            batch_connection.close()

            with self.assertRaises(AssertionError):
                load_resolved_training_batches(tokenized_db.name, batch_db.name)


if __name__ == "__main__":
    unittest.main()
