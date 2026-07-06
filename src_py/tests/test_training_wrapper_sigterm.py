from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src_py.train.cli_args import (
    TrainingRequestArgs,
    TrainingWrapperLaunchArgs,
    model_to_cli_args,
    model_to_json_bytes,
)


class TestTrainingWrapperSigterm(unittest.TestCase):
    def _run_wrapper_and_collect_events(
        self, temp_dir: str
    ) -> tuple[list[dict[str, object]], int, str]:
        trajectory_path = Path(temp_dir) / "training.sqlite"
        trajectory_path.write_bytes(b"sqlite")
        wrapper_log_path = Path(temp_dir) / "training_wrapper.log"

        launch_args = TrainingWrapperLaunchArgs(
            num_gpus=1,
            trajectory_sqlite_path=str(trajectory_path),
            hf_model_name="Qwen/Qwen3.5-4B",
            wrapper_log_path=str(wrapper_log_path),
            test_sleep_secs=30,
        )
        training_request = TrainingRequestArgs(
            training_plan="lora",
            advantage_clip=1.0,
            learning_rate=1e-4,
            weight_decay=0.01,
            grad_accum_steps=1,
            log_time_interval=5.0,
            checkpoint_save_time_interval=60.0,
            seed=7,
            training_time=60.0,
            num_iterations_limit=10,
            artifact_root_dir="/tmp",
            model_cli_name="qwen35_4b",
            config_nickname="sigterm_test",
            epoch=1,
            model_parent_dir="/tmp/results/qwen35_4b/sigterm_test/epoch_1",
            checkpoints_parent_dir="/tmp/results/qwen35_4b/sigterm_test/epoch_1",
            final_model_output_parent_dir="/tmp/results/qwen35_4b/sigterm_test/epoch_2",
            training_summary_parent_dir="/tmp/results/qwen35_4b/sigterm_test/summary",
            adam_fp32=False,
        )
        command = [
            sys.executable,
            "-m",
            "src_py.wrappers.training_wrapper",
            *model_to_cli_args(launch_args),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        process.stdin.write(model_to_json_bytes(training_request))
        process.stdin.close()
        process.stdin = None

        time.sleep(1.5)
        process.send_signal(signal.SIGTERM)
        _stdout_bytes, stderr_bytes = process.communicate(timeout=20)
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        log_text = wrapper_log_path.read_text(encoding="utf-8")
        parsed_events: list[dict[str, object]] = []
        for line in log_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed_events.append(json.loads(line))

        self.assertTrue(
            parsed_events,
            f"no events emitted; stderr={stderr_text}; log={log_text}",
        )
        running_events = [
            event
            for event in parsed_events
            if event.get("type") == "status" and event.get("status") == "running"
        ]
        self.assertTrue(
            running_events, f"missing running status event: {parsed_events}"
        )
        running_message = str(running_events[-1].get("message") or "")
        self.assertIn(str(wrapper_log_path), running_message)
        return parsed_events, process.returncode, stderr_text

    def test_hpc_wrapper_emits_cancelled_result_on_sigterm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wrapper_sigterm_test_") as temp_dir:
            parsed_events, returncode, _stderr_text = (
                self._run_wrapper_and_collect_events(temp_dir)
            )
            cancelling = [
                event
                for event in parsed_events
                if event.get("type") == "status" and event.get("status") == "cancelling"
            ]
            self.assertTrue(
                cancelling, f"missing cancelling status event: {parsed_events}"
            )

            results = [
                event for event in parsed_events if event.get("type") == "result"
            ]
            self.assertTrue(results, f"missing result event: {parsed_events}")
            last = results[-1]
            self.assertFalse(last.get("ok", True), f"expected failed result: {last}")
            self.assertEqual(
                last.get("error_code"), "CANCELLED_BY_SIGNAL", f"bad result: {last}"
            )
            self.assertNotEqual(returncode, 0)

    def test_modal_wrapper_emits_cancelled_result_on_sigterm(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="wrapper_sigterm_test_modal_"
        ) as temp_dir:
            parsed_events, returncode, _stderr_text = (
                self._run_wrapper_and_collect_events(temp_dir)
            )
            cancelling = [
                event
                for event in parsed_events
                if event.get("type") == "status" and event.get("status") == "cancelling"
            ]
            self.assertTrue(
                cancelling, f"missing cancelling status event: {parsed_events}"
            )

            results = [
                event for event in parsed_events if event.get("type") == "result"
            ]
            self.assertTrue(results, f"missing result event: {parsed_events}")
            last = results[-1]
            self.assertFalse(last.get("ok", True), f"expected failed result: {last}")
            self.assertEqual(
                last.get("error_code"), "CANCELLED_BY_SIGNAL", f"bad result: {last}"
            )
            self.assertNotEqual(returncode, 0)


if __name__ == "__main__":
    unittest.main()
