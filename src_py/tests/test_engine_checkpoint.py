import tempfile
import unittest
from pathlib import Path

import torch

from src_py.train.engine import (
    _load_checkpoint,
    _read_latest_checkpoint_pointer,
    _reset_oneshot_epoch_resume_state,
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
                checkpoint_tag="checkpoints",
                training_plan="lora",
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
                checkpoint_tag="checkpoints",
                training_plan="lora",
                adam_fp32=False,
            )

            self.assertTrue(torch.allclose(expected_weight, model.weight.detach()))
            self.assertTrue(torch.allclose(expected_bias, model.bias.detach()))
            self.assertEqual(3, resumed.global_step)
            self.assertEqual(1, resumed.next_iteration_index)
            self.assertEqual(2, resumed.next_batch_cursor)
            self.assertEqual(0, resumed.accumulation_step)
            self.assertEqual("checkpoints", _read_latest_checkpoint_pointer(output_dir))

    def test_save_and_load_checkpoint_roundtrip_ddp_plan(self) -> None:
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
                checkpoint_tag="checkpoints",
                training_plan="ddp",
                global_step=4,
                next_iteration_index=2,
                next_batch_cursor=1,
                accumulation_step=0,
            )

            with torch.no_grad():
                model.weight.add_(5.0)
                model.bias.sub_(5.0)

            resumed = _load_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                checkpoint_tag="checkpoints",
                training_plan="ddp",
                adam_fp32=False,
            )

            self.assertTrue(torch.allclose(expected_weight, model.weight.detach()))
            self.assertTrue(torch.allclose(expected_bias, model.bias.detach()))
            self.assertEqual(4, resumed.global_step)
            self.assertEqual(2, resumed.next_iteration_index)
            self.assertEqual(1, resumed.next_batch_cursor)
            self.assertEqual(0, resumed.accumulation_step)
            self.assertEqual("checkpoints", _read_latest_checkpoint_pointer(output_dir))

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
                    training_plan="lora",
                    global_step=1,
                    next_iteration_index=0,
                    next_batch_cursor=1,
                    accumulation_step=1,
                )

    def test_checkpoint_plan_aliases_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = torch.nn.Linear(4, 3)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                checkpoint_tag="checkpoints",
                training_plan="lora_current",
                global_step=1,
                next_iteration_index=0,
                next_batch_cursor=0,
                accumulation_step=0,
            )
            resumed = _load_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                checkpoint_tag="checkpoints",
                training_plan="lora",
                adam_fp32=False,
            )
            self.assertEqual(1, resumed.global_step)

    def test_reset_oneshot_epoch_resume_state_resets_time_and_iteration_budget(
        self,
    ) -> None:
        from src_py.train.engine import ResumeState

        resume_state = ResumeState(
            global_step=17,
            next_iteration_index=3,
            next_batch_cursor=11,
            accumulation_step=0,
            next_sample_index=29,
            next_batch_size=4,
            adaptive_velocity=0.25,
            adaptive_throughput_ema=5.5,
            adaptive_best_throughput_ema=6.5,
            adaptive_memory_utilization_ema=0.8,
            adaptive_previous_tokens_per_sample=321.0,
            adaptive_next_batch_size_float=4.5,
            elapsed_training_time_sec=600.0,
            samples_trained=123,
            samples_available=456,
            max_average_absolute_advantage=2.0,
            min_average_absolute_advantage=0.5,
            median_average_absolute_advantage=1.0,
        )

        reset_state = _reset_oneshot_epoch_resume_state(resume_state)

        self.assertEqual(17, reset_state.global_step)
        self.assertEqual(0, reset_state.next_iteration_index)
        self.assertEqual(11, reset_state.next_batch_cursor)
        self.assertEqual(29, reset_state.next_sample_index)
        self.assertEqual(4, reset_state.next_batch_size)
        self.assertAlmostEqual(0.0, reset_state.elapsed_training_time_sec)
        self.assertAlmostEqual(5.5, reset_state.adaptive_throughput_ema)
        self.assertAlmostEqual(4.5, reset_state.adaptive_next_batch_size_float)

    def test_resolve_resume_checkpoint_tag_latest_and_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            latest_pointer = output_dir / "latest_checkpoint.txt"

            self.assertEqual("", _resolve_resume_checkpoint_tag(output_dir, "none"))
            self.assertEqual("", _resolve_resume_checkpoint_tag(output_dir, "auto"))

            latest_pointer.write_text("checkpoints\n", encoding="utf-8")
            self.assertEqual(
                "checkpoints", _resolve_resume_checkpoint_tag(output_dir, "latest")
            )
            self.assertEqual(
                "checkpoints", _resolve_resume_checkpoint_tag(output_dir, "auto")
            )
            self.assertEqual(
                "checkpoints", _resolve_resume_checkpoint_tag(output_dir, "checkpoints")
            )


if __name__ == "__main__":
    unittest.main()
