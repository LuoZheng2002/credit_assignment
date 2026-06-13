from __future__ import annotations

from dataclasses import dataclass

import torch

from .data_sqlite import TrainingSampleTokenized

IGNORE_LABEL = -100


@dataclass(frozen=True)
class CollatedTrainingBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    advantages: torch.Tensor


def collate_training_samples(
    samples: list[TrainingSampleTokenized],
    pad_token_id: int,
) -> CollatedTrainingBatch:
    assert len(samples) > 0, "samples cannot be empty"
    assert pad_token_id >= 0, "pad_token_id must be non-negative"

    max_length = max(sample.input_length for sample in samples)
    assert max_length > 0, "max sequence length must be positive"

    input_id_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    advantages: list[list[float]] = []

    for sample in samples:
        assert len(sample.input_ids) == len(sample.labels), (
            "input_ids and labels must align"
        )
        assert sample.input_length == len(sample.input_ids), "input_length mismatch"
        assert sample.input_length > 0, "sample must contain tokens"
        assert len(sample.token_advantages) == sample.input_length, (
            "token_advantages and input_ids lengths must match"
        )

        pad_count = max_length - sample.input_length
        assert pad_count >= 0, "pad_count cannot be negative"

        input_id_rows.append(sample.input_ids + [pad_token_id] * pad_count)
        label_rows.append(sample.labels + [IGNORE_LABEL] * pad_count)
        attention_rows.append([1] * sample.input_length + [0] * pad_count)
        advantages.append(sample.token_advantages + [0.0] * pad_count)

    input_ids_tensor = torch.tensor(input_id_rows, dtype=torch.long)
    labels_tensor = torch.tensor(label_rows, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_rows, dtype=torch.long)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

    assert input_ids_tensor.ndim == 2, "input_ids tensor must be rank-2"
    assert labels_tensor.shape == input_ids_tensor.shape, (
        "labels tensor shape must match input_ids"
    )
    assert attention_mask_tensor.shape == input_ids_tensor.shape, (
        "attention mask shape must match input_ids"
    )
    assert advantages_tensor.shape == input_ids_tensor.shape, (
        "advantages tensor shape must match input_ids"
    )

    return CollatedTrainingBatch(
        input_ids=input_ids_tensor,
        labels=labels_tensor,
        attention_mask=attention_mask_tensor,
        advantages=advantages_tensor,
    )
