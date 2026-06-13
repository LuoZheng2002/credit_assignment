import unittest

from src_py.train.engine import _is_nonfinite_logits_exception


class TestNonfiniteLogitsException(unittest.TestCase):
    def test_detects_expected_assertion_message(self) -> None:
        self.assertTrue(
            _is_nonfinite_logits_exception(AssertionError("logits must be finite"))
        )

    def test_ignores_other_errors(self) -> None:
        self.assertFalse(_is_nonfinite_logits_exception(AssertionError("other")))
        self.assertFalse(_is_nonfinite_logits_exception(RuntimeError("other")))


if __name__ == "__main__":
    unittest.main()
