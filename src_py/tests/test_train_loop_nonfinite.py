import unittest
from typing import Any

import torch

from src_py.train.train_loop import (
    _DISTRIBUTED_CONTROL_RUN,
    _DISTRIBUTED_CONTROL_SKIP,
    _DISTRIBUTED_CONTROL_STOP,
    TrainingLoopClock,
    _assert_pre_step_finite,
    _extract_nonfinite_trace_suffix,
    _flush_partial_gradients,
    _plan_distributed_step_control,
    trace_first_nonfinite_backward_signal,
)


class _BadGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        assert len(grad_outputs) == 1, "expected a single grad output"
        grad_output = grad_outputs[0]
        return (grad_output * torch.tensor(float("nan"), device=grad_output.device),)


class _BadGradBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _BadGradFn.apply(x)


class _ToyBackwardNanModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.bad = _BadGradBlock()
        self.proj = torch.nn.Linear(4, 8)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        hidden = self.bad(hidden)
        return {"logits": self.proj(hidden)}


class TestTrainLoopNonfinite(unittest.TestCase):
    def test_rejects_nonfinite_gradients_before_optimizer_step(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.weight.grad = torch.zeros_like(model.weight)
        model.bias.grad = torch.zeros_like(model.bias)
        model.weight.grad[0, 0] = float("nan")

        with self.assertRaises(AssertionError) as context:
            _assert_pre_step_finite(model, optimizer, clipped_grad_norm=1.0)

        message = str(context.exception)
        self.assertIn("gradients must be finite", message)
        self.assertIn("nan_count=1", message)
        self.assertIn("inf_count=0", message)
        self.assertIn("first_nonfinite=parameter:weight", message)

    def test_rejects_nonfinite_optimizer_state_before_optimizer_step(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        inputs = torch.randn(4, 3)
        loss = model(inputs).sum()
        loss.backward()
        optimizer.step()

        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)

        optimizer.state[model.weight]["exp_avg"][0, 0] = float("inf")

        with self.assertRaises(AssertionError) as context:
            _assert_pre_step_finite(model, optimizer, clipped_grad_norm=1.0)

        message = str(context.exception)
        self.assertIn("optimizer_state must be finite", message)
        self.assertIn("nan_count=0", message)
        self.assertIn("inf_count=1", message)
        self.assertIn("first_nonfinite=parameter:weight", message)
        self.assertIn("state_key=exp_avg", message)

    def test_rejects_nonfinite_grad_norm_before_optimizer_step(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.weight.grad = torch.zeros_like(model.weight)
        model.bias.grad = torch.zeros_like(model.bias)

        with self.assertRaises(AssertionError) as context:
            _assert_pre_step_finite(model, optimizer, clipped_grad_norm=float("nan"))

        message = str(context.exception)
        self.assertIn("grad_norm must be finite", message)
        self.assertIn("nan_count=1", message)
        self.assertIn("inf_count=0", message)

    def test_extract_nonfinite_trace_suffix_returns_embedded_trace(self) -> None:
        message = (
            "gradients must be finite: nan_count=1 inf_count=0"
            " accumulation_micro_step=6"
            " nonfinite_backward_trace=1 status=first_nonfinite_backward_signal module_name=bad"
        )
        self.assertEqual(
            " nonfinite_backward_trace=1 status=first_nonfinite_backward_signal module_name=bad",
            _extract_nonfinite_trace_suffix(message),
        )

    def test_trace_first_nonfinite_backward_signal_reports_bad_module(self) -> None:
        model = _ToyBackwardNanModel()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, 2, 3]], dtype=torch.long)
        advantages = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)

        trace = trace_first_nonfinite_backward_signal(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            advantages=advantages,
            advantage_clip=3.0,
            grad_accum_steps=1,
        )

        self.assertIn("nonfinite_backward_trace=1", trace)
        self.assertIn("status=first_nonfinite_backward_signal", trace)
        self.assertIn("module_name=bad", trace)
        self.assertIn("module_type=_BadGradBlock", trace)
        self.assertIn("module_grad_output=all_finite", trace)
        self.assertIn("module_grad_input=path=grad_input[0]", trace)
        self.assertIn("parameter_name=embed.weight", trace)
        self.assertIn("parameter_grad=path=grad", trace)
        self.assertIn("nan_count=12", trace)

    def test_plan_distributed_step_control_stops_all_ranks_when_time_budget_expires(
        self,
    ) -> None:
        clock = TrainingLoopClock(
            training_time=600.0,
            resumed_elapsed_training_time_sec=0.0,
            run_start_time=0.0,
            training_end_time=float("-inf"),
            last_checkpoint_save_time=0.0,
            last_log_time=0.0,
            last_master_progress_time=0.0,
        )

        control = _plan_distributed_step_control(
            clock=clock,
            iteration_index=2,
            num_iterations_limit=5,
            sample_count=100,
            global_sample_cursor=32,
            world_size=2,
        )

        self.assertEqual(_DISTRIBUTED_CONTROL_STOP, control.opcode)
        self.assertEqual(0, control.requested_batch_size)
        self.assertEqual(32, control.global_sample_cursor)
        self.assertEqual(2, control.iteration_index)

    def test_plan_distributed_step_control_skips_when_not_enough_samples(
        self,
    ) -> None:
        clock = TrainingLoopClock(
            training_time=600.0,
            resumed_elapsed_training_time_sec=0.0,
            run_start_time=0.0,
            training_end_time=float("inf"),
            last_checkpoint_save_time=0.0,
            last_log_time=0.0,
            last_master_progress_time=0.0,
        )

        control = _plan_distributed_step_control(
            clock=clock,
            iteration_index=4,
            num_iterations_limit=10,
            sample_count=9,
            global_sample_cursor=8,
            world_size=2,
        )

        self.assertEqual(_DISTRIBUTED_CONTROL_SKIP, control.opcode)
        self.assertEqual(0, control.requested_batch_size)
        self.assertEqual(0, control.global_sample_cursor)
        self.assertEqual(5, control.iteration_index)

    def test_plan_distributed_step_control_runs_with_fixed_batch_size(self) -> None:
        clock = TrainingLoopClock(
            training_time=600.0,
            resumed_elapsed_training_time_sec=0.0,
            run_start_time=0.0,
            training_end_time=float("inf"),
            last_checkpoint_save_time=0.0,
            last_log_time=0.0,
            last_master_progress_time=0.0,
        )

        control = _plan_distributed_step_control(
            clock=clock,
            iteration_index=1,
            num_iterations_limit=10,
            sample_count=21,
            global_sample_cursor=9,
            world_size=2,
        )

        self.assertEqual(_DISTRIBUTED_CONTROL_RUN, control.opcode)
        self.assertEqual(1, control.requested_batch_size)
        self.assertEqual(9, control.global_sample_cursor)
        self.assertEqual(1, control.iteration_index)

    def test_flush_partial_gradients_discards_distributed_unsynced_microbatch(
        self,
    ) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)
        before_weight = model.weight.detach().clone()
        before_bias = model.bias.detach().clone()

        accumulation_step, global_step, clipped_grad_norm, current_lr = (
            _flush_partial_gradients(
                model=model,
                optimizer=optimizer,
                accumulation_step=1,
                global_step=7,
                max_grad_norm=1.0,
                base_learning_rate=1e-3,
                lr_warmup_steps=0,
                lr_min_scale=0.1,
                is_distributed=True,
            )
        )

        self.assertEqual(0, accumulation_step)
        self.assertEqual(7, global_step)
        self.assertIsNone(clipped_grad_norm)
        self.assertAlmostEqual(1e-3, current_lr)
        self.assertTrue(torch.allclose(before_weight, model.weight.detach()))
        self.assertTrue(torch.allclose(before_bias, model.bias.detach()))
        self.assertIsNone(model.weight.grad)
        self.assertIsNone(model.bias.grad)


if __name__ == "__main__":
    unittest.main()
