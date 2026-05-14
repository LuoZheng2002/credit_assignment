from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


IGNORE_LABEL = -100
STD_EPS = 1e-6


@dataclass(frozen=True)
class AdvantageWeightedLossOutput:
    loss: torch.Tensor
    stats: dict[str, float]


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced
    return tensor


def _global_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert values.ndim == 1, "values must be rank-1"
    assert values.numel() > 0, "values cannot be empty"

    values_fp32 = values.to(torch.float32)
    local_count = torch.tensor([values_fp32.numel()], device=values.device, dtype=torch.float32)
    local_sum = values_fp32.sum().unsqueeze(0)
    local_sum_sq = (values_fp32 * values_fp32).sum().unsqueeze(0)

    global_count = _all_reduce_sum(local_count)
    global_sum = _all_reduce_sum(local_sum)
    global_sum_sq = _all_reduce_sum(local_sum_sq)

    assert global_count.item() > 0.0, "global count must be positive"

    mean = global_sum / global_count
    var = global_sum_sq / global_count - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var + STD_EPS)
    return mean.squeeze(0), std.squeeze(0)


def _global_weighted_mean(local_sum: torch.Tensor, local_count: torch.Tensor) -> torch.Tensor:
    assert local_sum.ndim == 0, "local_sum must be scalar"
    assert local_count.ndim == 0, "local_count must be scalar"

    global_sum = _all_reduce_sum(local_sum.unsqueeze(0)).squeeze(0)
    global_count = _all_reduce_sum(local_count.unsqueeze(0)).squeeze(0)

    assert global_count.item() > 0.0, "global_count must be positive"
    return global_sum / global_count


def compute_advantage_weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    advantage_clip: float,
) -> AdvantageWeightedLossOutput:
    assert logits.ndim == 3, "logits must be [batch, seq_len, vocab]"
    assert labels.ndim == 2, "labels must be [batch, seq_len]"
    assert advantages.ndim == 1, "advantages must be [batch]"

    batch_size, seq_len, vocab_size = logits.shape
    assert labels.shape[0] == batch_size, "labels batch size must match logits"
    assert labels.shape[1] == seq_len, "labels seq_len must match logits"
    assert advantages.shape[0] == batch_size, "advantages batch size must match logits"
    assert seq_len >= 2, "seq_len must be >= 2 to support causal shift"
    assert vocab_size >= 2, "vocab_size must be >= 2"
    assert torch.isfinite(logits).all(), "logits must be finite"
    assert torch.isfinite(advantages).all(), "advantages must be finite"
    assert advantage_clip > 0.0, "advantage_clip must be > 0"

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    token_losses = F.cross_entropy(
        shifted_logits.view(-1, vocab_size),
        shifted_labels.view(-1),
        ignore_index=IGNORE_LABEL,
        reduction="none",
    ).view(batch_size, seq_len - 1)

    supervised_mask = shifted_labels.ne(IGNORE_LABEL)
    supervised_counts = supervised_mask.sum(dim=1)
    assert torch.all(supervised_counts > 0), "every sample must have at least one supervised token"

    supervised_counts_fp = supervised_counts.to(token_losses.dtype)
    per_sample_unweighted = (token_losses * supervised_mask).sum(dim=1) / supervised_counts_fp

    advantage_mean, advantage_std = _global_mean_std(advantages)
    normalized_advantages = (advantages.to(torch.float32) - advantage_mean) / advantage_std
    clipped_advantages = torch.clamp(normalized_advantages, min=-advantage_clip, max=advantage_clip)

    weighted_loss = (per_sample_unweighted * clipped_advantages.to(per_sample_unweighted.dtype)).mean()
    assert torch.isfinite(weighted_loss), "weighted loss must be finite"

    local_batch_count = torch.tensor(float(batch_size), device=logits.device, dtype=torch.float32)
    local_supervised_tokens = supervised_counts.to(torch.float32).sum()

    global_unweighted_ce = _global_weighted_mean(
        per_sample_unweighted.detach().to(torch.float32).sum(),
        local_batch_count,
    )
    global_weighted_loss = _global_weighted_mean(
        (per_sample_unweighted.detach().to(torch.float32) * clipped_advantages.detach()).sum(),
        local_batch_count,
    )
    global_adv_raw_mean = _global_weighted_mean(
        advantages.detach().to(torch.float32).sum(),
        local_batch_count,
    )
    global_adv_norm_mean = _global_weighted_mean(
        clipped_advantages.detach().to(torch.float32).sum(),
        local_batch_count,
    )
    global_tokens_per_sample = _global_weighted_mean(local_supervised_tokens, local_batch_count)

    stats = {
        "loss_weighted": float(global_weighted_loss.item()),
        "loss_unweighted_ce": float(global_unweighted_ce.item()),
        "advantage_raw_mean": float(global_adv_raw_mean.item()),
        "advantage_raw_std": float(advantage_std.item()),
        "advantage_normalized_mean": float(global_adv_norm_mean.item()),
        "supervised_tokens_per_sample": float(global_tokens_per_sample.item()),
        "batch_size_per_rank": float(batch_size),
    }

    return AdvantageWeightedLossOutput(loss=weighted_loss, stats=stats)
