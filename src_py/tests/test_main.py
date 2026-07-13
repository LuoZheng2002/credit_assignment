import tempfile
import unittest
from pathlib import Path

from src_py.train.cli_args import (
    TrainingHyperparametersRequest,
    TrainingRequestArgs,
    TrainProcessLaunchArgs,
)
from src_py.train.main import _load_train_config


class TestMain(unittest.TestCase):
    def _orchestration_request(self, **overrides: object) -> TrainingRequestArgs:
        payload: dict[str, object] = {
            "hyperparameters": {
                "lora_or_full": "lora",
                "distributed_strategy": "ddp",
                "advantage_clip": 3.0,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "grad_accum_steps": 1,
                "log_time_interval": 1,
                "seed": 42,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
            },
            "training_mode": {
                "type": "orchestration",
                "epoch": 3,
                "training_time": 10,
                "input_model_parent_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_3",
                "output_model_parent_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_4",
                "training_summary_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_3",
            },
            "model_cli_name": "qwen35_08b",
            "config_nickname": "run_a",
            "num_iterations_limit": 100,
        }
        payload.update(overrides)  # type: ignore[arg-type]
        return TrainingRequestArgs.model_validate(payload)

    def test_load_train_config_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trajectory_path = Path(tmp_dir) / "training_trajectories.msgpack"
            trajectory_path.write_bytes(b"msgpack")
            launch_args = TrainProcessLaunchArgs(
                training_trajectory_path=str(trajectory_path),
                training_request_json_path="/tmp/request.json",
            )
            request = self._orchestration_request()
            config = _load_train_config(launch_args, request)
            self.assertEqual("lora", config.lora_or_full)
            self.assertEqual("ddp", config.distributed_strategy)
            self.assertEqual(10, int(config.training_time))
            self.assertEqual(
                str(trajectory_path), config.training_trajectory_path
            )
            self.assertEqual(
                "/tmp/storage_root/results/qwen35_08b/run_a/epoch_3",
                config.model_parent_dir,
            )
            self.assertEqual(
                "/tmp/storage_root/results/qwen35_08b/run_a/epoch_3",
                config.training_summary_parent_dir,
            )
            self.assertEqual(
                "/tmp/storage_root/results/qwen35_08b/run_a/epoch_4",
                config.final_model_output_parent_dir,
            )

    def test_load_train_config_uses_epoch_zero_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trajectory_path = Path(tmp_dir) / "training_trajectories.msgpack"
            trajectory_path.write_bytes(b"msgpack")
            launch_args = TrainProcessLaunchArgs(
                training_trajectory_path=str(trajectory_path),
                training_request_json_path="/tmp/request.json",
            )
            request = self._orchestration_request(
                training_mode={
                    "type": "orchestration",
                    "epoch": 0,
                    "training_time": 10,
                    "input_model_parent_dir": "/tmp/storage_root/results/qwen35_08b",
                    "output_model_parent_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_1",
                    "training_summary_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_0",
                },
            )
            config = _load_train_config(launch_args, request)
            self.assertEqual(
                "/tmp/storage_root/results/qwen35_08b",
                config.model_parent_dir,
            )
            self.assertEqual(
                "/tmp/storage_root/results/qwen35_08b/run_a/epoch_0",
                config.training_summary_parent_dir,
            )

    def test_load_train_config_requires_uploaded_msgpack(self) -> None:
        launch_args = TrainProcessLaunchArgs(
            training_trajectory_path="/tmp/missing_training_trajectories.msgpack",
            training_request_json_path="/tmp/request.json",
        )
        request = self._orchestration_request()
        with self.assertRaises(AssertionError):
            _load_train_config(launch_args, request)

    def test_load_train_config_keeps_explicit_paths_when_hpc_root_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trajectory_path = Path(tmp_dir) / "training_trajectories.msgpack"
            trajectory_path.write_bytes(b"msgpack")
            launch_args = TrainProcessLaunchArgs(
                training_trajectory_path=str(trajectory_path),
                training_request_json_path="/tmp/request.json",
            )
            request = self._orchestration_request(
                hpc_training_root_dir="/tmp/hpc_volume_root",
                training_mode={
                    "type": "orchestration",
                    "epoch": 1,
                    "training_time": 10,
                    "input_model_parent_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_1",
                    "output_model_parent_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_2",
                    "training_summary_dir": "/tmp/storage_root/results/qwen35_08b/run_a/epoch_1",
                },
            )
            config = _load_train_config(launch_args, request)
            self.assertEqual(
                "/tmp/storage_root/results/qwen35_08b/run_a/epoch_1",
                config.model_parent_dir,
            )


if __name__ == "__main__":
    unittest.main()
