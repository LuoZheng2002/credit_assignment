import tempfile
import unittest
from pathlib import Path

from src_py.train.cli_args import TrainingRequestArgs, TrainProcessLaunchArgs
from src_py.train.main import _load_train_config


class TestMain(unittest.TestCase):
    def _training_request(self, **overrides: object) -> TrainingRequestArgs:
        payload: dict[str, object] = {
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
        payload.update(overrides)
        return TrainingRequestArgs.model_validate(payload)

    def test_load_train_config_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trajectory_path = Path(tmp_dir) / "training_trajectories.sqlite"
            trajectory_path.write_bytes(b"sqlite")
            launch_args = TrainProcessLaunchArgs(
                training_trajectory_sqlite_path=str(trajectory_path),
            )
            request = self._training_request()
            config = _load_train_config(launch_args, request)
            self.assertEqual("lora", config.training_plan)
            self.assertEqual("auto", config.resume_checkpoint_tag)
            self.assertEqual(10, int(config.training_time))
            self.assertEqual(
                str(trajectory_path), config.training_trajectory_sqlite_path
            )

    def test_load_train_config_uses_derived_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trajectory_path = Path(tmp_dir) / "training_trajectories.sqlite"
            trajectory_path.write_bytes(b"sqlite")
            launch_args = TrainProcessLaunchArgs(
                training_trajectory_sqlite_path=str(trajectory_path),
            )
            request = self._training_request(epoch=0)
            config = _load_train_config(launch_args, request)
            self.assertIn(
                "/tmp/storage_root/results/qwen35_08b", config.model_parent_dir
            )
            self.assertIn(
                "/tmp/storage_root/results/qwen35_08b/run_a/epoch_0",
                config.checkpoints_parent_dir,
            )

    def test_load_train_config_requires_uploaded_sqlite(self) -> None:
        launch_args = TrainProcessLaunchArgs(
            training_trajectory_sqlite_path="/tmp/missing_training_trajectories.sqlite",
        )
        request = self._training_request(epoch=2)
        with self.assertRaises(AssertionError):
            _load_train_config(launch_args, request)

    def test_load_train_config_prefers_hpc_training_root_dir_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trajectory_path = Path(tmp_dir) / "training_trajectories.sqlite"
            trajectory_path.write_bytes(b"sqlite")
            launch_args = TrainProcessLaunchArgs(
                training_trajectory_sqlite_path=str(trajectory_path),
            )
            request = self._training_request(
                artifact_root_dir="/tmp/storage_root",
                hpc_training_root_dir="/tmp/hpc_volume_root",
                epoch=1,
            )
            config = _load_train_config(launch_args, request)
            self.assertIn(
                "/tmp/hpc_volume_root/results/qwen35_08b", config.model_parent_dir
            )


if __name__ == "__main__":
    unittest.main()
