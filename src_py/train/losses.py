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
    local_count = torch.tensor(
        [values_fp32.numel()], device=values.device, dtype=torch.float32
    )
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


def _tensor_nonfinite_counts(tensor: torch.Tensor) -> tuple[int, int]:
    tensor_fp32 = tensor.to(torch.float32)
    nan_count = int(torch.isnan(tensor_fp32).sum().item())
    inf_count = int(torch.isinf(tensor_fp32).sum().item())
    return nan_count, inf_count


def _assert_tensor_finite(tensor: torch.Tensor, tensor_name: str) -> None:
    nan_count, inf_count = _tensor_nonfinite_counts(tensor)
    assert nan_count == 0 and inf_count == 0, (
        f"{tensor_name} must be finite: nan_count={nan_count} inf_count={inf_count}"
    )


def _global_weighted_mean(
    local_sum: torch.Tensor, local_count: torch.Tensor
) -> torch.Tensor:
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
    ref_logits: torch.Tensor | None = None,
    kl_penalty_coefficient: float = 0.0,
) -> AdvantageWeightedLossOutput:
    assert logits.ndim == 3, "logits must be [batch, seq_len, vocab]"
    assert labels.ndim == 2, "labels must be [batch, seq_len]"
    assert advantages.ndim == 2, "advantages must be [batch, seq_len]"

    batch_size, seq_len, vocab_size = logits.shape
    assert labels.shape[0] == batch_size, "labels batch size must match logits"
    assert labels.shape[1] == seq_len, "labels seq_len must match logits"
    assert advantages.shape[0] == batch_size, "advantages batch size must match logits"
    assert advantages.shape[1] == seq_len, "advantages seq_len must match logits"
    assert seq_len >= 2, "seq_len must be >= 2 to support causal shift"
    assert vocab_size >= 2, "vocab_size must be >= 2"
    assert advantage_clip > 0.0, "advantage_clip must be > 0"

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_advantages = advantages[:, 1:].contiguous()

    _assert_tensor_finite(shifted_logits, "logits")
    _assert_tensor_finite(shifted_advantages, "advantages")

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
    advantage_mean, advantage_std = _global_mean_std(raw_supervised_advantages)
    per_token_advantages = torch.clamp(
        raw_supervised_advantages.to(torch.float32),
        min=-advantage_clip,
        max=advantage_clip,
    )
    weighted_loss = (
        token_losses.masked_select(supervised_mask).to(torch.float32)
        * per_token_advantages
    ).sum() / supervised_count_fp
    _assert_tensor_finite(weighted_loss, "weighted_loss")

    total_loss = weighted_loss
    kl_div_mean: torch.Tensor | None = None
    if ref_logits is not None and kl_penalty_coefficient > 0.0:
        shifted_ref_logits = ref_logits[:, :-1, :].contiguous()
        assert shifted_ref_logits.shape == shifted_logits.shape, (
            "ref_logits shape must match logits shape"
        )
        _assert_tensor_finite(shifted_ref_logits, "ref_logits")

        curr_log_probs = F.log_softmax(shifted_logits.to(torch.float32), dim=-1)
        ref_log_probs = F.log_softmax(shifted_ref_logits.to(torch.float32), dim=-1)
        ref_probs = F.softmax(shifted_ref_logits.to(torch.float32), dim=-1)

        per_token_kl = (ref_probs * (ref_log_probs - curr_log_probs)).sum(dim=-1)
        kl_div_mean = (
            per_token_kl.masked_select(supervised_mask).to(torch.float32).sum()
            / supervised_count_fp
        )
        _assert_tensor_finite(kl_div_mean, "kl_div")
        total_loss = total_loss + kl_penalty_coefficient * kl_div_mean

    _assert_tensor_finite(total_loss, "total_loss")

    local_batch_count = torch.tensor(
        float(batch_size), device=logits.device, dtype=torch.float32
    )
    local_supervised_tokens = supervised_mask.to(torch.float32).sum()

    global_unweighted_ce = _global_weighted_mean(
        token_losses.masked_select(supervised_mask).to(torch.float32).sum(),
        local_supervised_tokens,
    )
    global_weighted_loss = _global_weighted_mean(
        (
            token_losses.masked_select(supervised_mask).to(torch.float32)
            * per_token_advantages
        ).sum(),
        local_supervised_tokens,
    )
    global_total_loss = _global_weighted_mean(
        total_loss.detach() * local_supervised_tokens,
        local_supervised_tokens,
    )
    global_adv_norm_mean = _global_weighted_mean(
        per_token_advantages.detach().to(torch.float32).sum(),
        local_supervised_tokens,
    )
    global_tokens_per_sample = _global_weighted_mean(
        local_supervised_tokens, local_batch_count
    )

    stats = {
        "loss_weighted": float(global_weighted_loss.item()),
        "loss_total": float(global_total_loss.item()),
        "loss_unweighted_ce": float(global_unweighted_ce.item()),
        "advantage_raw_mean": float(advantage_mean.item()),
        "advantage_raw_std": float(advantage_std.item()),
        "advantage_normalized_mean": float(global_adv_norm_mean.item()),
        "supervised_tokens_per_sample": float(global_tokens_per_sample.item()),
        "batch_size_per_rank": float(batch_size),
    }
    if kl_div_mean is not None:
        global_kl_div = _global_weighted_mean(
            kl_div_mean.detach() * local_supervised_tokens,
            local_supervised_tokens,
        )
        stats["kl_div"] = float(global_kl_div.item())

    return AdvantageWeightedLossOutput(loss=total_loss, stats=stats)


@dataclass(frozen=True)
class SftLossOutput:
    loss: torch.Tensor
    stats: dict[str, float]


def compute_sft_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> SftLossOutput:
    """Standard causal LM cross-entropy loss for SFT training.

    Only computes loss on non-IGNORE_LABEL tokens.
    """
    assert logits.ndim == 3, "logits must be [batch, seq_len, vocab]"
    assert labels.ndim == 2, "labels must be [batch, seq_len]"

    batch_size, seq_len, vocab_size = logits.shape
    assert labels.shape[0] == batch_size, "labels batch size must match logits"
    assert labels.shape[1] == seq_len, "labels seq_len must match logits"
    assert seq_len >= 2, "seq_len must be >= 2 to support causal shift"
    assert vocab_size >= 2, "vocab_size must be >= 2"

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    _assert_tensor_finite(shifted_logits, "logits")

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

    supervised_count_fp = supervised_count.to(token_losses.dtype)
    loss = (
        token_losses.masked_select(supervised_mask).to(torch.float32).sum()
        / supervised_count_fp
    )
    _assert_tensor_finite(loss, "sft_loss")

    stats = {
        "loss_ce": float(loss.item()),
        "supervised_tokens_per_sample": float(supervised_count_fp.item() / batch_size),
        "batch_size_per_rank": float(batch_size),
    }

    return SftLossOutput(loss=loss, stats=stats)
