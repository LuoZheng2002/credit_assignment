import unittest

import torch
import torch.nn.functional as F

from src_py.train.losses import IGNORE_LABEL, compute_advantage_weighted_causal_lm_loss


class TestAdvantageWeightedLoss(unittest.TestCase):
    def test_matches_per_sample_weighting_and_not_token_raw_weighting(self) -> None:
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
        advantages = torch.tensor([0.0, 2.0], dtype=torch.float32)

        output = compute_advantage_weighted_causal_lm_loss(
            logits=logits,
            labels=labels,
            advantages=advantages,
            advantage_clip=10.0,
        )

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        token_losses = F.cross_entropy(
            shifted_logits.reshape(-1, vocab_size),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_LABEL,
            reduction="none",
        ).reshape(batch_size, seq_len - 1)

        mask = shifted_labels.ne(IGNORE_LABEL)
        token_counts = mask.sum(dim=1)
        self.assertTrue(torch.all(token_counts > 0).item())

        per_sample = (token_losses * mask).sum(dim=1) / token_counts.to(token_losses.dtype)
        adv_norm = (advantages - advantages.mean()) / advantages.std(unbiased=False)

        expected_per_sample_weighted = (per_sample * adv_norm).mean()

        raw_weighted = (token_losses * mask * adv_norm.unsqueeze(1)).sum() / mask.sum()

        self.assertTrue(
            torch.allclose(output.loss.detach(), expected_per_sample_weighted, atol=1e-6),
            "Loss must match per-sample weighting",
        )
        self.assertFalse(
            torch.allclose(output.loss.detach(), raw_weighted, atol=1e-6),
            "Loss should not match per-token raw weighting when token counts differ",
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
        advantages = torch.tensor([0.0, 100.0], dtype=torch.float32)

        output = compute_advantage_weighted_causal_lm_loss(
            logits=logits,
            labels=labels,
            advantages=advantages,
            advantage_clip=0.25,
        )

        self.assertLessEqual(abs(output.stats["advantage_normalized_mean"]), 0.25)
        self.assertGreater(output.stats["advantage_raw_std"], 0.0)

    def test_raises_when_sample_has_no_supervised_token(self) -> None:
        logits = torch.randn(2, 4, 5, dtype=torch.float32)
        labels = torch.tensor(
            [
                [IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL],
                [IGNORE_LABEL, 1, 2, 3],
            ],
            dtype=torch.long,
        )
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

        with self.assertRaises(AssertionError):
            compute_advantage_weighted_causal_lm_loss(
                logits=logits,
                labels=labels,
                advantages=advantages,
                advantage_clip=3.0,
            )


if __name__ == "__main__":
    unittest.main()
