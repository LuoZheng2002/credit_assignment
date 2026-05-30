import tempfile
import unittest
from pathlib import Path

from src_py.train.main_from_config import _load_train_config_from_toml


def _to_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        else:
            lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


class TestMainFromConfig(unittest.TestCase):
    def test_load_train_config_from_toml_success(self) -> None:
        payload = {
            "training_plan": "lora_current",
            "model_parent_dir": "/tmp/models/qwen35_08b_parent",
            "training_trajectory_sqlite_path": "/tmp/training_trajectories.sqlite",
            "checkpoints_parent_dir": "/tmp/run_a",
            "final_model_output_parent_dir": "/tmp/final_model_hf_parent",
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "training_time": 10,
            "grad_accum_steps": 1,
            "log_time_interval": 1,
            "checkpoint_save_time_interval": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules_csv": "q_proj,k_proj",
            "resume_checkpoint_tag": "auto",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(_to_toml(payload), encoding="utf-8")
            config = _load_train_config_from_toml(str(config_path))
            self.assertEqual("lora_current", config.training_plan)
            self.assertEqual("auto", config.resume_checkpoint_tag)
            self.assertEqual(10, int(config.training_time))

    def test_load_train_config_from_toml_requires_training_time(self) -> None:
        payload = {
            "training_plan": "lora_current",
            "model_parent_dir": "/tmp/models/qwen35_08b_parent",
            "training_trajectory_sqlite_path": "/tmp/training_trajectories.sqlite",
            "checkpoints_parent_dir": "/tmp/run_a",
            "final_model_output_parent_dir": "/tmp/final_model_hf_parent",
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "training_time": 10,
            "grad_accum_steps": 1,
            "log_time_interval": 1,
            "checkpoint_save_time_interval": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules_csv": "q_proj,k_proj",
            "resume_checkpoint_tag": "auto",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(_to_toml(payload), encoding="utf-8")
            config = _load_train_config_from_toml(str(config_path))
            self.assertEqual(10, int(config.training_time))

    def test_load_train_config_from_toml_rejects_missing_or_extra_keys(self) -> None:
        payload = {
            "training_plan": "lora_current",
            "model_parent_dir": "/tmp/models/qwen35_08b_parent",
            "training_trajectory_sqlite_path": "/tmp/training_trajectories.sqlite",
            "checkpoints_parent_dir": "/tmp/run_a",
            "final_model_output_parent_dir": "/tmp/final_model_hf_parent",
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "training_time": 10,
            "grad_accum_steps": 1,
            "log_time_interval": 1,
            "checkpoint_save_time_interval": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules_csv": "q_proj,k_proj",
            "seed": 42,
            "unexpected": 123,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad_config.toml"
            config_path.write_text(_to_toml(payload), encoding="utf-8")
            with self.assertRaises(AssertionError):
                _load_train_config_from_toml(str(config_path))


if __name__ == "__main__":
    unittest.main()
