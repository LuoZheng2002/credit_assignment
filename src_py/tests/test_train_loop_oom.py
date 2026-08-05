import unittest

from src_py.train.train_loop import _plan_cuda_oom


class TestTrainLoopOom(unittest.TestCase):
    def test_oom_stops_training_without_iteration_limit_retry(self) -> None:
        plan = _plan_cuda_oom(iteration_index=0)
        self.assertTrue(plan.stop_training)
        self.assertEqual(1, plan.next_iteration_index)


if __name__ == "__main__":
    unittest.main()
