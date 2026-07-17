import unittest

from src_py.train.train_loop import _plan_cuda_oom


class TestTrainLoopOom(unittest.TestCase):
    def test_oom_before_iteration_limit_restarts_next_iteration(self) -> None:
        plan = _plan_cuda_oom(iteration_index=0, num_iterations_limit=3)
        self.assertFalse(plan.stop_training)
        self.assertEqual(1, plan.next_iteration_index)

    def test_oom_at_iteration_limit_stops_training(self) -> None:
        plan = _plan_cuda_oom(iteration_index=1, num_iterations_limit=2)
        self.assertTrue(plan.stop_training)
        self.assertEqual(2, plan.next_iteration_index)


if __name__ == "__main__":
    unittest.main()
