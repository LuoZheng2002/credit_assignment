import unittest

from src_py.train.engine import (
    _is_nonfinite_logits_exception,
    _parse_nonfinite_tensor_exception,
)


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

    def test_ignores_other_errors(self) -> None:
        self.assertFalse(_is_nonfinite_logits_exception(AssertionError("other")))
        self.assertFalse(_is_nonfinite_logits_exception(RuntimeError("other")))
        self.assertIsNone(_parse_nonfinite_tensor_exception(AssertionError("other")))
        self.assertIsNone(_parse_nonfinite_tensor_exception(RuntimeError("other")))


if __name__ == "__main__":
    unittest.main()
