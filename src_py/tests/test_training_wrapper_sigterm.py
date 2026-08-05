from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src_py.training_config_models import (
    TrainingHyperparametersRequest,
    TrainingRequestArgs,
    TrainingModeOneShot,
    model_to_json_bytes,
)
from src_py.wrappers.training_wrapper import TrainingWrapperStdinArgs


class TestTrainingWrapperSigterm(unittest.TestCase):
    def _run_wrapper_and_collect_events(
        self, temp_dir: str
    ) -> tuple[list[dict[str, object]], int, str]:
        trajectory_path = Path(temp_dir) / "training.msgpack"
        trajectory_path.write_bytes(b"msgpack")
        wrapper_log_path = Path(temp_dir) / "training_wrapper.log"

        training_request = TrainingRequestArgs(
            hyperparameters=TrainingHyperparametersRequest(
                lora_or_full="lora",
                distributed_strategy="ddp",
                advantage_clip=1.0,
                learning_rate=1e-4,
                weight_decay=0.01,
                use_adam_state=True,
                use_lr_warmup=True,
                grad_accum_steps=1,
                log_time_interval=5.0,
                lr_warmup_steps=100,
                seed=7,
            ),
            training_mode=TrainingModeOneShot(
                type="oneshot",
                per_epoch_training_time=60.0,
                num_oneshot_epochs=0,
                model_output_root="",
                training_summary_dir="/tmp/results/qwen35_4b/sigterm_test/epoch_1",
                base_model_parent_dir="/tmp/results/qwen35_4b/sigterm_test/epoch_1",
            ),
            num_iterations_limit=10,
            training_trajectory_len_cutoff=4096,
            model_cli_name="qwen35_4b",
            config_nickname="sigterm_test",
        )
        stdin_args = TrainingWrapperStdinArgs(
            training_config=training_request,
            num_gpus=1,
            trajectory_path=str(trajectory_path),
            hf_model_name="Qwen/Qwen3.5-4B",
            wrapper_log_path=str(wrapper_log_path),
            test_sleep_secs=30,
        )
        command = [
            sys.executable,
            "-m",
            "src_py.wrappers.training_wrapper",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        process.stdin.write(model_to_json_bytes(stdin_args))
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
