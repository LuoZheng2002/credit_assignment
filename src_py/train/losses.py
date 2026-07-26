from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_LABEL = -100
STD_EPS = 1e-6
POLICY_RATIO_CLIP = 0.2
LOG_RATIO_CLAMP = 20.0


@dataclass(frozen=True)
class AdvantageWeightedLossOutput:
    loss: torch.Tensor
    stats: dict[str, float]


def _local_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert values.ndim == 1, "values must be rank-1"
    assert values.numel() > 0, "values cannot be empty"

    mean = values.mean()
    var = torch.clamp(values.var(unbiased=False), min=0.0)
    std = torch.sqrt(var + STD_EPS)
    return mean, std


def _tensor_nonfinite_counts(tensor: torch.Tensor) -> tuple[int, int]:
    if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
        return 0, 0
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    return nan_count, inf_count


def _assert_tensor_finite(tensor: torch.Tensor, tensor_name: str) -> None:
    nan_count, inf_count = _tensor_nonfinite_counts(tensor)
    assert nan_count == 0 and inf_count == 0, (
        f"{tensor_name} must be finite: nan_count={nan_count} inf_count={inf_count}"
    )


def _local_weighted_mean(
    local_sum: torch.Tensor, local_count: torch.Tensor
) -> torch.Tensor:
    assert local_sum.ndim == 0, "local_sum must be scalar"
    assert local_count.ndim == 0, "local_count must be scalar"
    assert local_count.item() > 0.0, "local_count must be positive"
    return local_sum / local_count



def compute_advantage_weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage_clip: float,
) -> AdvantageWeightedLossOutput:
    assert logits.ndim == 3, "logits must be [batch, seq_len, vocab]"
    assert labels.ndim == 2, "labels must be [batch, seq_len]"
    assert advantages.ndim == 2, "advantages must be [batch, seq_len]"
    assert old_logprobs.ndim == 2, "old_logprobs must be [batch, seq_len]"

    batch_size, seq_len, vocab_size = logits.shape
    assert labels.shape[0] == batch_size, "labels batch size must match logits"
    assert labels.shape[1] == seq_len, "labels seq_len must match logits"
    assert advantages.shape[0] == batch_size, "advantages batch size must match logits"
    assert advantages.shape[1] == seq_len, "advantages seq_len must match logits"
    assert old_logprobs.shape[0] == batch_size, (
        "old_logprobs batch size must match logits"
    )
    assert old_logprobs.shape[1] == seq_len, (
        "old_logprobs seq_len must match logits"
    )
    assert seq_len >= 2, "seq_len must be >= 2 to support causal shift"
    assert vocab_size >= 2, "vocab_size must be >= 2"
    assert advantage_clip > 0.0, "advantage_clip must be > 0"

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_advantages = advantages[:, 1:].contiguous()
    shifted_old_logprobs = old_logprobs[:, 1:].contiguous()

    _assert_tensor_finite(shifted_logits, "logits")
    _assert_tensor_finite(shifted_advantages, "advantages")
    _assert_tensor_finite(shifted_old_logprobs, "old_logprobs")

    token_losses = F.cross_entropy(
        shifted_logits.view(-1, vocab_size),
        shifted_labels.view(-1),
        ignore_index=IGNORE_LABEL,
        reduction="none",
    ).view(batch_size, seq_len - 1)

    supervised_mask = shifted_labels.ne(IGNORE_LABEL)
    supervised_count = supervised_mask.sum()
    assert supervised_count.item() > 0, (
        "batch must contain at least one supervised token"
    )

    _assert_tensor_finite(token_losses, "token_losses")

    supervised_count_fp = supervised_count.to(token_losses.dtype)
    raw_supervised_advantages = shifted_advantages.masked_select(supervised_mask)
    supervised_old_logprobs = shifted_old_logprobs.masked_select(supervised_mask)
    advantage_mean, advantage_std = _local_mean_std(raw_supervised_advantages)
    per_token_advantages = torch.clamp(
        raw_supervised_advantages,
        min=-advantage_clip,
        max=advantage_clip,
    )
    supervised_token_losses = token_losses.masked_select(supervised_mask)
    supervised_new_logprobs = -supervised_token_losses
    log_ratios = (supervised_new_logprobs - supervised_old_logprobs).clamp(
        min=-LOG_RATIO_CLAMP,
        max=LOG_RATIO_CLAMP,
    )
    policy_ratios = torch.exp(log_ratios)
    clipped_policy_ratios = torch.clamp(
        policy_ratios,
        min=1.0 - POLICY_RATIO_CLIP,
        max=1.0 + POLICY_RATIO_CLIP,
    )
    unclipped_surrogate = policy_ratios * per_token_advantages
    clipped_surrogate = clipped_policy_ratios * per_token_advantages
    surrogate = torch.minimum(unclipped_surrogate, clipped_surrogate)
    policy_loss = -surrogate.sum() / supervised_count_fp
    _assert_tensor_finite(policy_loss, "policy_loss")

    total_loss = policy_loss

    _assert_tensor_finite(total_loss, "total_loss")

    local_batch_count = torch.tensor(
        float(batch_size), device=logits.device, dtype=torch.bfloat16
    )
    local_supervised_tokens = supervised_mask.to(torch.bfloat16).sum()

    local_unweighted_ce = _local_weighted_mean(
        token_losses.masked_select(supervised_mask).sum(),
        local_supervised_tokens,
    )
    local_policy_loss = _local_weighted_mean(policy_loss.detach() * local_supervised_tokens, local_supervised_tokens)
    local_mean_ratio = _local_weighted_mean(
        policy_ratios.detach().sum(),
        local_supervised_tokens,
    )
    local_clip_fraction = _local_weighted_mean(
        (
            (policy_ratios.detach() < 1.0 - POLICY_RATIO_CLIP)
            | (policy_ratios.detach() > 1.0 + POLICY_RATIO_CLIP)
        )
        .to(torch.bfloat16)
        .sum(),
        local_supervised_tokens,
    )
    local_total_loss = _local_weighted_mean(
        total_loss.detach() * local_supervised_tokens,
        local_supervised_tokens,
    )
    local_adv_norm_mean = _local_weighted_mean(
        per_token_advantages.detach().sum(),
        local_supervised_tokens,
    )
    local_tokens_per_sample = _local_weighted_mean(
        local_supervised_tokens, local_batch_count
    )

    stats = {
        "loss_weighted": float(local_policy_loss.item()),
        "loss_policy_surrogate": float(local_policy_loss.item()),
        "loss_total": float(local_total_loss.item()),
        "loss_unweighted_ce": float(local_unweighted_ce.item()),
        "policy_ratio_mean": float(local_mean_ratio.item()),
        "policy_ratio_clip_fraction": float(local_clip_fraction.item()),
        "policy_ratio_clip": float(POLICY_RATIO_CLIP),
        "advantage_raw_mean": float(advantage_mean.item()),
        "advantage_raw_std": float(advantage_std.item()),
        "advantage_normalized_mean": float(local_adv_norm_mean.item()),
        "supervised_tokens_per_sample": float(local_tokens_per_sample.item()),
        "batch_size_per_rank": float(batch_size),
    }

    return AdvantageWeightedLossOutput(loss=total_loss, stats=stats)

