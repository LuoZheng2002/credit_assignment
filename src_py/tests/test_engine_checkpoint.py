import tempfile
import unittest
from pathlib import Path

import torch

from src_py.train.engine import (
    _load_checkpoint,
    _read_latest_checkpoint_pointer,
    _resolve_resume_checkpoint_tag,
    _save_checkpoint,
)


class TestEngineCheckpoint(unittest.TestCase):
    def test_save_and_load_checkpoint_roundtrip_lora_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = torch.nn.Linear(4, 3)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            inputs = torch.randn(2, 4)
            loss = model(inputs).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            expected_weight = model.weight.detach().clone()
            expected_bias = model.bias.detach().clone()

            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                checkpoint_tag="step_3",
                training_plan="lora_current",
                global_step=3,
                next_iteration_index=1,
                next_batch_cursor=2,
                accumulation_step=0,
            )

            with torch.no_grad():
                model.weight.add_(10.0)
                model.bias.sub_(10.0)

            resumed = _load_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                checkpoint_tag="step_3",
                training_plan="lora_current",
            )

            self.assertTrue(torch.allclose(expected_weight, model.weight.detach()))
            self.assertTrue(torch.allclose(expected_bias, model.bias.detach()))
            self.assertEqual(3, resumed.global_step)
            self.assertEqual(1, resumed.next_iteration_index)
            self.assertEqual(2, resumed.next_batch_cursor)
            self.assertEqual(0, resumed.accumulation_step)
            self.assertEqual("step_3", _read_latest_checkpoint_pointer(output_dir))

    def test_save_checkpoint_rejects_partial_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            with self.assertRaises(AssertionError):
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    output_dir=output_dir,
                    checkpoint_tag="bad",
                    training_plan="lora_current",
                    global_step=1,
                    next_iteration_index=0,
                    next_batch_cursor=1,
                    accumulation_step=1,
                )

    def test_resolve_resume_checkpoint_tag_latest_and_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            latest_pointer = output_dir / "latest_checkpoint.txt"

            self.assertEqual("", _resolve_resume_checkpoint_tag(output_dir, "none"))
            self.assertEqual("", _resolve_resume_checkpoint_tag(output_dir, "auto"))

            latest_pointer.write_text("global_step_12\n", encoding="utf-8")
            self.assertEqual("global_step_12", _resolve_resume_checkpoint_tag(output_dir, "latest"))
            self.assertEqual("global_step_12", _resolve_resume_checkpoint_tag(output_dir, "auto"))
            self.assertEqual("final", _resolve_resume_checkpoint_tag(output_dir, "final"))


if __name__ == "__main__":
    unittest.main()
