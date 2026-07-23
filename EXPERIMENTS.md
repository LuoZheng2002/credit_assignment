# Finished Experiment Results

This file tracks completed or materially informative training experiments for the credit-assignment project. Accuracy values are validation averages unless otherwise stated.

## Summary

- The most reliable positive signal so far is Qwen2.5 no-tool GRPO with single-GPU LoRA.
- TreeMAPPO no-tool has a positive signal at LoRA rank 16 and at rank 32 with a low learning rate, but collapses at rank 32 with the original high learning rate.
- Mistral no-tool LoRA currently collapses after the first validated epoch and is not a main-paper path.
- Qwen34 completed training, but trained-epoch validation is blocked by vLLM `/update_model` reliability, so it is not yet a finished accuracy result.

## Successful Experiments

### Qwen2.5 No-Tool GRPO, LoRA Rank 16

- Config: `config/oneshot_train/qwen25_train_grpo_notool_lora.toml`
- Training mode: no-tool GRPO, single-GPU LoRA, rank 16, learning rate `5e-5`.
- Validation status: completed through 10 epochs.
- Accuracy:
  - base epoch 0: `0.6487`
  - best trained epoch: epoch 2, `0.6687`
  - final epoch 10: `0.6593`
- Improvement:
  - best gain: `+0.0200`
  - final gain: `+0.0107`
- Conclusion: stable positive signal, but gains are modest and not monotonic across epochs.

### Qwen2.5 No-Tool GRPO, LoRA Rank 32

- Config: `config/oneshot_train/qwen25_train_grpo_notool_lora_r32.toml`
- Training mode: no-tool GRPO, single-GPU LoRA, rank 32, learning rate `5e-5`.
- Validation status: completed through 10 epochs.
- Accuracy:
  - base epoch 0: `0.6503`
  - best trained epoch: epoch 10, `0.6703`
  - final epoch 10: `0.6703`
- Improvement:
  - best gain: `+0.0200`
  - final gain: `+0.0200`
- Conclusion: stable positive GRPO anchor; rank 32 does not collapse in the GRPO setting and is currently the cleanest no-tool baseline.

### Qwen2.5 No-Tool TreeMAPPO, LoRA Rank 16, Early Validation

- Config: `config/oneshot_train/qwen25_train_tree_notool_lora_r16_10ep_15m.toml`
- Training mode: no-tool TreeMAPPO, single-GPU LoRA, rank 16, learning rate `5e-5`.
- Validation status: completed through epoch 2.
- Accuracy:
  - base epoch 0: `0.6631`
  - epoch 1: `0.6861`
  - epoch 2: `0.6802`
- Improvement:
  - best gain: `+0.0230`
  - latest validated gain: `+0.0170`
- Conclusion: promising early TreeMAPPO signal, stronger than the GRPO rank-16/rank-32 gains in the best epoch, but this run did not validate all requested epochs.

### Qwen2.5 No-Tool TreeMAPPO, LoRA Rank 16, Five-Epoch Rerun

- Config: `config/oneshot_train/qwen25_train_tree_notool_lora_r16_5ep_15m.toml`
- Training mode: no-tool TreeMAPPO, single-GPU LoRA, rank 16, learning rate `5e-5`.
- Validation status: partial; epoch 1 validated.
- Accuracy:
  - base epoch 0: `0.6782`
  - epoch 1: `0.6965`
- Improvement:
  - validated gain: `+0.0183`
- Conclusion: independently reproduces a positive first-epoch TreeMAPPO rank-16 signal, but later epochs still need reliable validation.

### Qwen2.5 No-Tool TreeMAPPO, LoRA Rank 32, Low Learning Rate

- Config: `config/oneshot_train/qwen25_train_tree_notool_lora_r32_lr1e6_10ep_15m.toml`
- Training mode: no-tool TreeMAPPO, single-GPU LoRA, rank 32, learning rate `1e-6`.
- Validation status: completed through 10 epochs.
- Accuracy:
  - base epoch 0: `0.6761`
  - best trained epoch: epoch 10, `0.6931`
  - final epoch 10: `0.6931`
- Improvement:
  - best gain: `+0.0170`
  - final gain: `+0.0170`
- Conclusion: lowering the learning rate prevents the rank-32 TreeMAPPO collapse and gives a modest positive signal.

## Failed Accuracy Experiments

### Qwen2.5 No-Tool TreeMAPPO, LoRA Rank 32, High Learning Rate

- Config: `config/oneshot_train/qwen25_train_tree_notool_lora_r32_10ep_15m.toml`
- Training mode: no-tool TreeMAPPO, single-GPU LoRA, rank 32, learning rate `5e-5`.
- Validation status: completed through 10 epochs.
- Accuracy:
  - base epoch 0: `0.6580`
  - epoch 1: `0.0000`
  - best trained epoch: epoch 4, `0.2823`
  - final epoch 10: `0.2122`
- Outcome: catastrophic accuracy collapse.
- Conclusion: TreeMAPPO rank 32 is unstable at `5e-5`; higher-rank LoRA requires a much smaller learning rate or additional stabilization.

### Mistral No-Tool GRPO, LoRA Rank 16 Safe Run

- Config: `config/oneshot_train/mistral_train_grpo_notool_lora_r16_5m_safe.toml`
- Training mode: no-tool GRPO, single-GPU LoRA, rank 16, learning rate `5e-5`, 5 minutes per epoch.
- Validation status: completed through epoch 4.
- Accuracy:
  - base epoch 0: `0.2068`
  - epoch 1: `0.2224`
  - epoch 2: `0.0022`
  - epoch 3: `0.0000`
  - epoch 4: `0.0044`
- Outcome: initial small epoch-1 improvement followed by near-zero collapse.
- Conclusion: Mistral LoRA is currently unstable under this recipe and should not be prioritized for the paper result until the collapse mechanism is understood.

## Incomplete Or Infrastructure-Blocked Experiments

### Qwen2.5 No-Tool GRPO, LoRA Rank 48, Low Learning Rate

- Config: `config/oneshot_train/qwen25_train_grpo_notool_lora_r48_lr1e6_3ep_15m.toml`
- Training mode: no-tool GRPO, single-GPU LoRA, rank 48, learning rate `1e-6`.
- Validation status: partial; validated through epoch 2, then blocked by vLLM `/update_model` timeout.
- Accuracy:
  - base epoch 0: `0.6729`
  - epoch 1: `0.6717`
  - epoch 2: `0.6747`
- Current interpretation: stable but no meaningful gain yet; not enough validated epochs for a final conclusion.

### Qwen34 No-Tool GRPO, LoRA Rank 32

- Config: `config/oneshot_train/qwen34_train_grpo_notool_lora_r32_6h.toml`
- Training mode: no-tool GRPO, single-GPU LoRA, rank 32, learning rate `5e-5`, 24 epochs.
- Training status: completed all 24 epochs.
- Validation status: base epoch validated; trained-epoch validation blocked by vLLM `/update_model` timeout.
- Accuracy:
  - base epoch 0: `0.7055`
- Current interpretation: training completed without OOM, but there is no trained-model accuracy conclusion yet.

## Practical Conclusions

- For immediate paper evidence, focus on Qwen2.5 no-tool GRPO rank 32 and TreeMAPPO rank 16 / rank 32 low-learning-rate comparisons.
- Treat large learning rates such as `5e-5` as unsafe for higher-rank TreeMAPPO LoRA unless validated otherwise.
- Do not use Mistral as the primary evidence path until the post-epoch-1 collapse is diagnosed.
- Fixing vLLM validation restart reliability is necessary before Qwen34 or rank-48 results can be considered complete.
