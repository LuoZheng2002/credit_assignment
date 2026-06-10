import tempfile
import unittest
from pathlib import Path

from src_py.train.main import _load_train_config


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


class TestMain(unittest.TestCase):
    def test_load_train_config_success(self) -> None:
        payload = {
            "training_plan": "lora",
            "artifact_root_dir": "/tmp/storage_root",
            "model_cli_name": "qwen35_08b",
            "config_nickname": "run_a",
            "epoch": 3,
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "training_time": 10,
            "num_iterations_limit": 100,
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
            job_dir = Path(tmp_dir) / "job_a"
            (job_dir / "input").mkdir(parents=True, exist_ok=True)
            (job_dir / "input" / "training_trajectories.sqlite").write_bytes(b"sqlite")
            config_path = job_dir / "train_request.toml"
            config_path.write_text(_to_toml(payload), encoding="utf-8")
            config = _load_train_config(str(job_dir))
            self.assertEqual("lora", config.training_plan)
            self.assertEqual("auto", config.resume_checkpoint_tag)
            self.assertEqual(10, int(config.training_time))
            self.assertTrue(
                config.training_trajectory_sqlite_path.endswith(
                    "training_trajectories.sqlite"
                )
            )

    def test_load_train_config_uses_derived_paths(self) -> None:
        payload = {
            "training_plan": "lora",
            "artifact_root_dir": "/tmp/storage_root",
            "model_cli_name": "qwen35_08b",
            "config_nickname": "run_a",
            "epoch": 0,
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "training_time": 10,
            "num_iterations_limit": 100,
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
            job_dir = Path(tmp_dir) / "job_b"
            (job_dir / "input").mkdir(parents=True, exist_ok=True)
            (job_dir / "input" / "training_trajectories.sqlite").write_bytes(b"sqlite")
            config_path = job_dir / "train_request.toml"
            config_path.write_text(_to_toml(payload), encoding="utf-8")
            config = _load_train_config(str(job_dir))
            self.assertIn(
                "/tmp/storage_root/results/qwen35_08b", config.model_parent_dir
            )
            self.assertIn(
                "/tmp/storage_root/results/qwen35_08b/run_a/epoch_0",
                config.checkpoints_parent_dir,
            )

    def test_load_train_config_requires_uploaded_sqlite(self) -> None:
        payload = {
            "training_plan": "lora",
            "artifact_root_dir": "/tmp/storage_root",
            "model_cli_name": "qwen35_08b",
            "config_nickname": "run_a",
            "epoch": 2,
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "training_time": 10,
            "num_iterations_limit": 100,
            "grad_accum_steps": 1,
            "log_time_interval": 1,
            "checkpoint_save_time_interval": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules_csv": "q_proj,k_proj",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_dir = Path(tmp_dir) / "job_missing_sqlite"
            job_dir.mkdir(parents=True, exist_ok=True)
            config_path = job_dir / "train_request.toml"
            config_path.write_text(_to_toml(payload), encoding="utf-8")
            with self.assertRaises(AssertionError):
                _load_train_config(str(job_dir))

    def test_load_train_config_prefers_hpc_training_root_dir_when_present(self) -> None:
        payload = {
            "training_plan": "lora",
            "artifact_root_dir": "/tmp/storage_root",
            "hpc_training_root_dir": "/tmp/hpc_volume_root",
            "model_cli_name": "qwen35_08b",
            "config_nickname": "run_a",
            "epoch": 1,
            "advantage_clip": 3.0,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "training_time": 10,
            "num_iterations_limit": 100,
            "grad_accum_steps": 1,
            "log_time_interval": 1,
            "checkpoint_save_time_interval": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules_csv": "q_proj,k_proj",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_dir = Path(tmp_dir) / "job_hpc_root"
            (job_dir / "input").mkdir(parents=True, exist_ok=True)
            (job_dir / "input" / "training_trajectories.sqlite").write_bytes(b"sqlite")
            config_path = job_dir / "train_request.toml"
            config_path.write_text(_to_toml(payload), encoding="utf-8")
            config = _load_train_config(str(job_dir))
            self.assertIn(
                "/tmp/hpc_volume_root/results/qwen35_08b", config.model_parent_dir
            )


if __name__ == "__main__":
    unittest.main()
