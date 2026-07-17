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
            elapsed_training_time_sec=600.0,
            samples_trained=123,
            samples_available=456,
            max_average_absolute_advantage=2.0,
            min_average_absolute_advantage=0.5,
            median_average_absolute_advantage=1.0,
            samples_trained_this_run=77,
            longest_non_oom_trajectory_length=1024,
            stopped_due_to_oom=True,
        )

        reset_state = _reset_oneshot_epoch_resume_state(resume_state)

        self.assertEqual(17, reset_state.global_step)
        self.assertEqual(0, reset_state.next_iteration_index)
        self.assertEqual(11, reset_state.next_batch_cursor)
        self.assertEqual(29, reset_state.next_sample_index)
        self.assertAlmostEqual(0.0, reset_state.elapsed_training_time_sec)
        self.assertEqual(123, reset_state.samples_trained)
        self.assertEqual(456, reset_state.samples_available)
        self.assertEqual(0, reset_state.samples_trained_this_run)
        self.assertEqual(0, reset_state.longest_non_oom_trajectory_length)
        self.assertFalse(reset_state.stopped_due_to_oom)



if __name__ == "__main__":
    unittest.main()
