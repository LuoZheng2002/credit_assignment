import unittest

import torch

from src_py.train.losses import (
    IGNORE_LABEL,
    POLICY_RATIO_CLIP,
    compute_advantage_weighted_causal_lm_loss,
)


class TestAdvantageWeightedLoss(unittest.TestCase):
    def test_matches_clipped_surrogate(self) -> None:
        torch.manual_seed(7)

        batch_size = 2
        seq_len = 5
        vocab_size = 3
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)

        labels = torch.tensor(
            [
                [IGNORE_LABEL, 0, 1, IGNORE_LABEL, IGNORE_LABEL],
                [IGNORE_LABEL, 1, 2, 1, 0],
            ],
            dtype=torch.long,
        )
        advantages = torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, -1.0, 2.0, -2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        old_logprobs = torch.full((batch_size, seq_len), -1.0, dtype=torch.float32)

        output = compute_advantage_weighted_causal_lm_loss(
            logits=logits,
            labels=labels,
            advantages=advantages,
            old_logprobs=old_logprobs,
            advantage_clip=10.0,
        )

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        shifted_advantages = advantages[:, 1:]
        shifted_old_logprobs = old_logprobs[:, 1:]

        mask = shifted_labels.ne(IGNORE_LABEL)
        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        safe_labels = shifted_labels.clamp_min(0)
        selected_new_logprobs = log_probs.gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        supervised_new_logprobs = selected_new_logprobs.masked_select(mask)
        supervised_old_logprobs = shifted_old_logprobs.masked_select(mask)
        supervised_advantages = shifted_advantages.masked_select(mask)
        ratios = torch.exp(supervised_new_logprobs - supervised_old_logprobs)
        clipped_ratios = torch.clamp(
            ratios,
            min=1.0 - POLICY_RATIO_CLIP,
            max=1.0 + POLICY_RATIO_CLIP,
        )
        expected = -torch.minimum(
            ratios * supervised_advantages,
            clipped_ratios * supervised_advantages,
        ).sum() / mask.sum()

        self.assertTrue(
            torch.allclose(output.loss.detach(), expected, atol=1e-6),
            "Loss must match PPO/GRPO clipped surrogate",
        )

    def test_advantage_clipping_applies(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]],
                [[-1.0, 2.0], [-1.0, 2.0], [-1.0, 2.0]],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [IGNORE_LABEL, 0, 0],
                [IGNORE_LABEL, 1, 1],
            ],
            dtype=torch.long,
        )
        advantages = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 100.0, -100.0],
            ],
            dtype=torch.float32,
        )
        old_logprobs = torch.zeros_like(advantages)

        output = compute_advantage_weighted_causal_lm_loss(
            logits=logits,
            labels=labels,
            advantages=advantages,
            old_logprobs=old_logprobs,
            advantage_clip=0.25,
        )

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        shifted_advantages = advantages[:, 1:].clamp(-0.25, 0.25)
        mask = shifted_labels.ne(IGNORE_LABEL)
        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        selected_new_logprobs = log_probs.gather(
            dim=-1,
            index=shifted_labels.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)
        ratios = torch.exp(selected_new_logprobs.masked_select(mask))
        clipped_ratios = torch.clamp(
            ratios,
            min=1.0 - POLICY_RATIO_CLIP,
            max=1.0 + POLICY_RATIO_CLIP,
        )
        supervised_advantages = shifted_advantages.masked_select(mask)
        expected = -torch.minimum(
            ratios * supervised_advantages,
            clipped_ratios * supervised_advantages,
        ).sum() / mask.sum()

        self.assertTrue(torch.allclose(output.loss.detach(), expected, atol=1e-6))
        self.assertLessEqual(abs(output.stats["advantage_normalized_mean"]), 0.25)
        self.assertGreater(output.stats["advantage_raw_std"], 0.0)

    def test_raises_when_batch_has_no_supervised_token(self) -> None:
        logits = torch.randn(2, 4, 5, dtype=torch.float32)
        labels = torch.tensor(
            [
                [IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL],
                [IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL],
            ],
            dtype=torch.long,
        )
        advantages = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        old_logprobs = torch.zeros_like(advantages)

        with self.assertRaises(AssertionError):
            compute_advantage_weighted_causal_lm_loss(
                logits=logits,
                labels=labels,
                advantages=advantages,
                old_logprobs=old_logprobs,
                advantage_clip=3.0,
            )

    def test_rejects_nan_or_inf_logits(self) -> None:
        labels = torch.tensor([[IGNORE_LABEL, 0]], dtype=torch.long)
        advantages = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        old_logprobs = torch.zeros_like(advantages)

        for value, nan_count, inf_count in [
            (float("nan"), 1, 0),
            (float("inf"), 0, 1),
            (float("-inf"), 0, 1),
        ]:
            logits = torch.tensor([[[value, 1.0], [0.0, 1.0]]], dtype=torch.float32)
            with self.assertRaises(AssertionError) as context:
                compute_advantage_weighted_causal_lm_loss(
                    logits=logits,
                    labels=labels,
                    advantages=advantages,
                    old_logprobs=old_logprobs,
                    advantage_clip=3.0,
                )
            self.assertIn("logits must be finite", str(context.exception))
            self.assertIn(f"nan_count={nan_count}", str(context.exception))
            self.assertIn(f"inf_count={inf_count}", str(context.exception))

    def test_rejects_nan_or_inf_advantages(self) -> None:
        logits = torch.tensor([[[0.0, 1.0], [0.5, -0.5]]], dtype=torch.float32)
        labels = torch.tensor([[IGNORE_LABEL, 0]], dtype=torch.long)
        old_logprobs = torch.tensor([[0.0, -0.1]], dtype=torch.float32)

        for value, nan_count, inf_count in [
            (float("nan"), 1, 0),
            (float("inf"), 0, 1),
            (float("-inf"), 0, 1),
        ]:
            advantages = torch.tensor([[0.0, value]], dtype=torch.float32)
            with self.assertRaises(AssertionError) as context:
                compute_advantage_weighted_causal_lm_loss(
                    logits=logits,
                    labels=labels,
                    advantages=advantages,
                    old_logprobs=old_logprobs,
                    advantage_clip=3.0,
                )
            self.assertIn("advantages must be finite", str(context.exception))
            self.assertIn(f"nan_count={nan_count}", str(context.exception))
            self.assertIn(f"inf_count={inf_count}", str(context.exception))

    def test_rejects_nan_or_inf_old_logprobs(self) -> None:
        logits = torch.tensor([[[0.0, 1.0], [0.5, -0.5]]], dtype=torch.float32)
        labels = torch.tensor([[IGNORE_LABEL, 0]], dtype=torch.long)
        advantages = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        for value, nan_count, inf_count in [
            (float("nan"), 1, 0),
            (float("inf"), 0, 1),
            (float("-inf"), 0, 1),
        ]:
            old_logprobs = torch.tensor([[0.0, value]], dtype=torch.float32)
            with self.assertRaises(AssertionError) as context:
                compute_advantage_weighted_causal_lm_loss(
                    logits=logits,
                    labels=labels,
                    advantages=advantages,
                    old_logprobs=old_logprobs,
                    advantage_clip=3.0,
                )
            self.assertIn("old_logprobs must be finite", str(context.exception))
            self.assertIn(f"nan_count={nan_count}", str(context.exception))
            self.assertIn(f"inf_count={inf_count}", str(context.exception))


if __name__ == "__main__":
    unittest.main()
