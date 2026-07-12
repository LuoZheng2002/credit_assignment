import unittest

from src_py.train.engine import (
    _reset_oneshot_epoch_resume_state,
)


class TestEngineCheckpoint(unittest.TestCase):
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



if __name__ == "__main__":
    unittest.main()
