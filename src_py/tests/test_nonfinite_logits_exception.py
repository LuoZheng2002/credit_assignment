import unittest

import torch

from src_py.train.engine import (
    _is_nonfinite_logits_exception,
    _parse_nonfinite_tensor_exception,
    trace_first_nonfinite_forward_module,
)


class _PassThroughBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _NanBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tensor(float("nan"), device=x.device, dtype=x.dtype)


class _ToyCausalLm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(32, 4)
        self.safe = _PassThroughBlock()
        self.bad = _NanBlock()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        hidden = self.safe(hidden)
        logits = self.bad(hidden)
        return {"logits": logits}


class TestNonfiniteLogitsException(unittest.TestCase):
    def test_detects_expected_assertion_message(self) -> None:
        self.assertTrue(
            _is_nonfinite_logits_exception(
                AssertionError("logits must be finite: nan_count=1 inf_count=0")
            )
        )

    def test_parses_nonfinite_tensor_details(self) -> None:
        self.assertEqual(
            ("logits", 2, 0),
            _parse_nonfinite_tensor_exception(
                AssertionError("logits must be finite: nan_count=2 inf_count=0")
            ),
        )
        self.assertEqual(
            ("advantages", 0, 3),
            _parse_nonfinite_tensor_exception(
                RuntimeError("advantages must be finite: nan_count=0 inf_count=3")
            ),
        )
        self.assertEqual(
            ("optimizer_state", 0, 1),
            _parse_nonfinite_tensor_exception(
                AssertionError(
                    "optimizer_state must be finite: nan_count=0 inf_count=1 first_nonfinite=parameter:weight state_key=exp_avg"
                )
            ),
        )

    def test_ignores_other_errors(self) -> None:
        self.assertFalse(_is_nonfinite_logits_exception(AssertionError("other")))
        self.assertFalse(_is_nonfinite_logits_exception(RuntimeError("other")))
        self.assertIsNone(_parse_nonfinite_tensor_exception(AssertionError("other")))
        self.assertIsNone(_parse_nonfinite_tensor_exception(RuntimeError("other")))

    def test_trace_first_nonfinite_forward_module_reports_first_bad_module(
        self,
    ) -> None:
        model = _ToyCausalLm()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        trace = trace_first_nonfinite_forward_module(
            model_engine=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        self.assertIn("status=first_nonfinite_output", trace)
        self.assertIn("module_name=bad", trace)
        self.assertIn("module_type=_NanBlock", trace)
        self.assertIn("module_input=all_finite", trace)
        self.assertIn("module_output=path=output", trace)
        self.assertIn("nan_count=12", trace)


if __name__ == "__main__":
    unittest.main()
