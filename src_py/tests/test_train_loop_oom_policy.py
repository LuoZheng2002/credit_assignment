import unittest

from src_py.train.train_loop import _should_fail_fast_on_cuda_oom


class TestTrainLoopCudaOomPolicy(unittest.TestCase):
    def test_single_gpu_never_fails_fast(self) -> None:
        self.assertFalse(
            _should_fail_fast_on_cuda_oom(
                distributed_strategy="single_gpu",
                world_size=1,
            )
        )

    def test_ddp_fails_fast_when_world_size_exceeds_one(self) -> None:
        self.assertTrue(
            _should_fail_fast_on_cuda_oom(
                distributed_strategy="ddp",
                world_size=4,
            )
        )

    def test_fsdp_fails_fast_when_world_size_exceeds_one(self) -> None:
        self.assertTrue(
            _should_fail_fast_on_cuda_oom(
                distributed_strategy="fsdp",
                world_size=4,
            )
        )

    def test_distributed_named_strategy_without_multi_rank_does_not_fail_fast(self) -> None:
        self.assertFalse(
            _should_fail_fast_on_cuda_oom(
                distributed_strategy="ddp",
                world_size=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
