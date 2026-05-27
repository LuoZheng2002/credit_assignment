import json
import tempfile
import unittest
from pathlib import Path

from src_py.train.main_from_config import _load_train_config_from_json


class TestMainFromConfig(unittest.TestCase):
    def test_load_train_config_from_json_success(self) -> None:
        payload = {
            "training_plan": "lora_current",
            "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "training_trajectory_sqlite_path": "/tmp/training_trajectories.sqlite",
            "batch_size": 8,
            "output_dir": "/tmp/out",
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "num_epochs": 1,
            "grad_accum_steps": 1,
            "log_interval_steps": 1,
            "save_interval_steps": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules_csv": "q_proj,k_proj",
            "resume_checkpoint_tag": "auto",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = _load_train_config_from_json(str(config_path))
            self.assertEqual("lora_current", config.training_plan)
            self.assertEqual("auto", config.resume_checkpoint_tag)

    def test_load_train_config_from_json_rejects_missing_or_extra_keys(self) -> None:
        payload = {
            "training_plan": "lora_current",
            "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "training_trajectory_sqlite_path": "/tmp/training_trajectories.sqlite",
            "batch_size": 8,
            "output_dir": "/tmp/out",
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "num_epochs": 1,
            "grad_accum_steps": 1,
            "log_interval_steps": 1,
            "save_interval_steps": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules_csv": "q_proj,k_proj",
            "seed": 42,
            "unexpected": 123,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AssertionError):
                _load_train_config_from_json(str(config_path))


if __name__ == "__main__":
    unittest.main()
