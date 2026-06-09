from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class TestTrainingWrapperSigterm(unittest.TestCase):
    def test_hpc_wrapper_emits_cancelled_result_on_sigterm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wrapper_sigterm_test_") as temp_dir:
            trajectory_path = Path(temp_dir) / "training.sqlite"
            trajectory_path.write_bytes(b"sqlite")

            command = [
                sys.executable,
                "-m",
                "src_py.wrappers.training_wrapper",
                "--backend",
                "hpc",
                "--num-gpus",
                "1",
                "--training-config-json",
                '{"model_cli_name":"qwen35_4b","config_nickname":"sigterm_test","epoch":1,"artifact_root_dir":"/tmp"}',
                "--trajectory-sqlite-path",
                str(trajectory_path),
                "--hf-model-name",
                "Qwen/Qwen3.5-4B",
                "--test-sleep-secs",
                "30",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            time.sleep(1.5)
            process.send_signal(signal.SIGTERM)
            stdout_text, stderr_text = process.communicate(timeout=20)

            parsed_events = []
            for line in stdout_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parsed_events.append(json.loads(line))

            self.assertTrue(parsed_events, f"no events emitted; stderr={stderr_text}")
            cancelling = [
                event
                for event in parsed_events
                if event.get("type") == "status" and event.get("status") == "cancelling"
            ]
            self.assertTrue(cancelling, f"missing cancelling status event: {parsed_events}")

            results = [event for event in parsed_events if event.get("type") == "result"]
            self.assertTrue(results, f"missing result event: {parsed_events}")
            last = results[-1]
            self.assertFalse(last.get("ok", True), f"expected failed result: {last}")
            self.assertEqual(last.get("error_code"), "CANCELLED_BY_SIGNAL", f"bad result: {last}")
            self.assertNotEqual(process.returncode, 0)

    def test_modal_wrapper_emits_cancelled_result_on_sigterm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wrapper_sigterm_test_modal_") as temp_dir:
            trajectory_path = Path(temp_dir) / "training.sqlite"
            trajectory_path.write_bytes(b"sqlite")

            command = [
                sys.executable,
                "-m",
                "src_py.wrappers.training_wrapper",
                "--backend",
                "modal",
                "--num-gpus",
                "1",
                "--training-config-json",
                '{"model_cli_name":"qwen35_4b","config_nickname":"sigterm_test","epoch":1,"artifact_root_dir":"/tmp"}',
                "--trajectory-sqlite-path",
                str(trajectory_path),
                "--hf-model-name",
                "Qwen/Qwen3.5-4B",
                "--test-sleep-secs",
                "30",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            time.sleep(1.5)
            process.send_signal(signal.SIGTERM)
            stdout_text, stderr_text = process.communicate(timeout=20)

            parsed_events = []
            for line in stdout_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parsed_events.append(json.loads(line))

            self.assertTrue(parsed_events, f"no events emitted; stderr={stderr_text}")
            cancelling = [
                event
                for event in parsed_events
                if event.get("type") == "status" and event.get("status") == "cancelling"
            ]
            self.assertTrue(cancelling, f"missing cancelling status event: {parsed_events}")

            results = [event for event in parsed_events if event.get("type") == "result"]
            self.assertTrue(results, f"missing result event: {parsed_events}")
            last = results[-1]
            self.assertFalse(last.get("ok", True), f"expected failed result: {last}")
            self.assertEqual(last.get("error_code"), "CANCELLED_BY_SIGNAL", f"bad result: {last}")
            self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
