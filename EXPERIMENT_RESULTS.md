# Experiment Results

This document records paper-relevant experiment outcomes in a table-first format. It intentionally separates:

- **held-out validation curves** used for model selection during training;
- **serious test runs** used for paper-facing evidence;
- **ablations and negative results** that constrain the claims.

Unless otherwise noted, training uses the one-shot pipeline, LoRA rank 32, learning rate `1e-6`, Adam state enabled, LR warmup enabled, vLLM inference, and held-out validation every available epoch or every 3 epochs in longer chunked runs.

## Current Interpretation

| Topic | Current conclusion | Evidence status |
|---|---|---|
| Qwen2.5 no-tool GRPO vs TreeMAPPO | Both improve slightly on validation; TreeMAPPO is competitive but not clearly better overall in serious tests yet. | Strongest completed long-run validation; serious 5-rollout tests complete for one GRPO and one Tree checkpoint. |
| Qwen2.5 tool GRPO vs TreeMAPPO | Tool setting improves over base; TreeMAPPO currently has a better single serious-test mean than GRPO in the older 5-epoch tests. | Long-run validation complete to 40 epochs; latest serious tests still need consolidation. |
| Qwen34 tool GRPO vs TreeMAPPO | GRPO is ahead on aggregate serious test, while TreeMAPPO wins some individual datasets. | Completed six-dataset serious tests. |
| Qwen34 no-tool GRPO vs TreeMAPPO | Both improve validation; TreeMAPPO has the higher best validation gain. | Serious tests are partial/incomplete by dataset. |
| Mistral/Gemma/Llama | Useful for robustness, but current runs are still completing or recovering from timeouts. | Do not use as primary paper evidence yet. |
| KL regularization | `kl_beta = 0.001` did not improve Qwen2.5 no-tool validation relative to non-KL Adam runs. | Partial to epoch 30. |
| SGD/no-warmup | Not effective in prior partial results. | Negative ablation; not primary. |

## Main Validation Results

Validation accuracy is the mean over the current held-out validation mixture in the corresponding run. Gains are absolute accuracy points, not relative percentages.

| Model | Setting | Method | Epochs trained | Validated epochs | Base | Best epoch | Best | Best gain | Last validated | Last gain | Status |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B | No-tool | GRPO | 70 | 0-21, then every 3 to 69 | 0.6430 | 51 | 0.6567 | +0.0137 | 69: 0.6497 | +0.0067 | Complete to epoch 69 summary |
| Qwen2.5-7B | No-tool | TreeMAPPO | 70 | 0-21, then every 3 to 69 | 0.6447 | 27 | 0.6553 | +0.0107 | 69: 0.6520 | +0.0073 | Complete to epoch 69 summary |
| Qwen2.5-7B | Tool | GRPO | 40 | every 3 epochs | 0.6113 | 6 | 0.6267 | +0.0153 | 39: 0.6157 | +0.0043 | Complete to epoch 39 summary |
| Qwen2.5-7B | Tool | TreeMAPPO | 40 | every 3 epochs | 0.6133 | 12 | 0.6287 | +0.0153 | 39: 0.6233 | +0.0100 | Complete to epoch 39 summary |
| Qwen34 | No-tool | GRPO | 10 | 0-10 | 0.7034 | 8 | 0.7147 | +0.0113 | 10: 0.7079 | +0.0045 | Complete validation |
| Qwen34 | No-tool | TreeMAPPO | 10 | 0-10 | 0.7132 | 3 | 0.7256 | +0.0124 | 10: 0.7114 | -0.0018 | Complete validation |
| Qwen34 | Tool | GRPO | 10 | 0-10 | 0.6753 | 9 | 0.6969 | +0.0216 | 10: 0.6922 | +0.0170 | Complete validation |
| Qwen34 | Tool | TreeMAPPO | 10 | 0-10 | 0.6623 | 1 | 0.6924 | +0.0302 | 10: 0.6880 | +0.0257 | Complete validation |

## Serious Test Results

Serious tests use broader held-out datasets. The Qwen2.5 no-tool GRPO/Tree tests below use 5 rollouts per question and report mean accuracy across rollouts where available.

### Qwen2.5 No-Tool: Base vs GRPO vs TreeMAPPO

| Model / checkpoint | DeepMath | MATH | NuminaMath | AMC2023 | CollegeMath | Gaokao 2024 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, epoch 0 | 0.6140 | 0.7370 | 0.6400 | 0.5000 | 0.7530 | 0.6538 | 0.6496 |
| GRPO, epoch 36, 5-rollout test | 0.6054 | 0.7350 | 0.5980 | 0.5900 | 0.7298 | 0.5692 | 0.6379 |
| TreeMAPPO, epoch 27, 5-rollout test | 0.6140 | 0.7332 | 0.5900 | 0.5950 | 0.7180 | 0.6077 | 0.6430 |

Interpretation: under this serious-test protocol, TreeMAPPO is slightly ahead of the matched GRPO checkpoint on aggregate, but both are below the single-run base aggregate. This argues for caution: validation improvements do not yet transfer cleanly to the broad serious-test suite.

### Qwen2.5 Tool: Base vs Early GRPO/Tree Tests

| Model / checkpoint | DeepMath | MATH | NuminaMath | AMC2023 | CollegeMath | Gaokao 2024 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, epoch 0 | 0.5860 | 0.6850 | 0.6200 | 0.4750 | 0.6847 | 0.5769 | 0.6046 |
| GRPO, epoch 5 | 0.5720 | 0.6920 | 0.6300 | 0.4750 | 0.7703 | 0.5385 | 0.6130 |
| TreeMAPPO, epoch 1 | 0.5920 | 0.6860 | 0.5900 | 0.5000 | 0.7770 | 0.6154 | 0.6267 |

Interpretation: in the older tool serious tests, TreeMAPPO has the best aggregate mean. Longer 40-epoch validation is available, but matching serious tests for the best long-run checkpoints should be preferred before making a strong paper claim.

### Qwen34 Tool: GRPO vs TreeMAPPO

| Model / checkpoint | DeepMath | MATH | NuminaMath | AMC2023 | CollegeMath | Gaokao 2024 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO, epoch 9 | 0.6540 | 0.7608 | 0.5918 | 0.5676 | 0.7823 | 0.6087 | 0.6609 |
| TreeMAPPO, epoch 1 | 0.6306 | 0.7497 | 0.6105 | 0.5897 | 0.7843 | 0.5217 | 0.6478 |

Interpretation: GRPO is currently better on aggregate. TreeMAPPO is better on AMC2023, CollegeMath, and NuminaMath, but loses enough on DeepMath, MATH, and Gaokao 2024 to fall behind overall.

### Qwen34 No-Tool Partial Serious Tests

| Model / checkpoint | DeepMath | MATH | NuminaMath | Mean over available datasets | Status |
|---|---:|---:|---:|---:|---|
| Base, epoch 0 | 0.5740 | 0.8924 | — | 0.7332 | Only two datasets recorded |
| GRPO, epoch 8 | 0.7000 | 0.8321 | — | 0.7661 | Only two datasets recorded |
| TreeMAPPO, epoch 3 | 0.6990 | 0.8340 | 0.6939 | 0.7423 | Three datasets recorded |

Interpretation: these are not directly comparable as full serious tests because the available dataset coverage differs.

## Ablation Results

### Qwen2.5 No-Tool Method Ablations

All rows use Qwen2.5 no-tool, LoRA rank 32, learning rate `1e-6`, Adam, and 5 epochs unless otherwise noted.

| Variant | What changes | Base | Best epoch | Best | Best gain | Final epoch | Final | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TEMPO-style branching | TreeMAPPO setup with spontaneous/TEMPO-like branching policy | 0.6367 | 4 | 0.6547 | +0.0180 | 5 | 0.6527 | Strong short-run validation gain; useful ablation baseline. |
| TreeRPO-style advantage | TreeMAPPO branching with TreeRPO-like credit calculation | 0.6487 | 0 | 0.6487 | +0.0000 | 5 | 0.6457 | No improvement in this run. |
| TreeRL advantage only | TreeMAPPO branching with TreeRL-style value backup advantage | 0.6390 | 3 | 0.6537 | +0.0147 | 5 | 0.6453 | Positive but less stable than TEMPO-style result. |
| TreeRL branching only | TreeRL-style branching with TreeMAPPO advantage | 0.6517 | 3 | 0.6567 | +0.0050 | 5 | 0.6483 | Weak, not clearly useful. |
| TreeMAPPO + KL beta 0.001 | Adds offline reference-logprob KL term | 0.6447 | 9 | 0.6510 | +0.0063 | 30 | 0.6500 | KL did not improve over non-KL TreeMAPPO. |
| GRPO + KL beta 0.001 | Adds offline reference-logprob KL term | 0.6460 | 24 | 0.6537 | +0.0077 | 30 | 0.6493 | KL did not improve over non-KL GRPO. |

### Branching Scale / Variant Tests

| Variant | Serious-test checkpoint | DeepMath | MATH | NuminaMath | AMC2023 | CollegeMath | Gaokao 2024 | Mean | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Branch-8 no-tool | epoch 3 | 0.5970 | 0.7510 | 0.6400 | 0.6000 | 0.8800 | 0.5769 | 0.6742 | Strong aggregate in single-run serious test; needs matched repeated test. |
| Branch-32 no-tool | epoch 5 | 0.6080 | 0.7420 | 0.6400 | 0.5500 | 0.8635 | 0.6154 | 0.6698 | Also strong; branch-count trend is not yet conclusive. |
| TreeMAPPO uncertainty + forced token | epoch 5 | 0.6040 | 0.7400 | 0.6600 | 0.5500 | 0.8757 | 0.6154 | 0.6742 | Competitive; useful for paper/code-consistency ablation. |
| Positive-advantage-only TreeMAPPO | epoch 5 | 0.6260 | 0.7440 | 0.6300 | 0.6500 | 0.8529 | 0.5769 | 0.6800 | Strong single-run aggregate, but treated as ablation not main method. |

## Non-Qwen Model Results

These runs are included for completeness even though they are not currently helpful to the main paper claim. The immediate follow-up plan is to reduce learning rate and LoRA rank further for Gemma and Mistral before treating failures as model-level conclusions.

### Gemma No-Tool Validation

| Model | Method | Setup | Validated epochs | Base | Best epoch | Best | Best gain | Last validated | Last gain | Interpretation |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Gemma | GRPO | older 5-epoch run, LoRA r32, lr `1e-6` | 0-5 | 0.5994 | 4 | 0.6092 | +0.0098 | 5: 0.6091 | +0.0097 | Mild positive validation signal. |
| Gemma | TreeMAPPO | older 5-epoch run, LoRA r32, lr `1e-6` | 0-5 | 0.6017 | 0 | 0.6017 | +0.0000 | 5: 0.5927 | -0.0090 | Tree run degraded under this setup. |
| Gemma | GRPO | chunked/resume run, LoRA r32, lr `1e-6` | 0, 3 | 0.5683 | 0 | 0.5683 | +0.0000 | 3: 0.5590 | -0.0093 | Current 30-epoch pipeline not yet favorable. |
| Gemma | TreeMAPPO | chunked/resume run, LoRA r32, lr `1e-6` | 0, 3 | 0.5717 | 0 | 0.5717 | +0.0000 | 3: 0.5597 | -0.0120 | Current 30-epoch pipeline not yet favorable. |

Interpretation: Gemma currently suggests either setup sensitivity or insufficiently conservative optimization. Future runs should lower LoRA rank and learning rate before excluding Gemma as a useful model.

### Mistral No-Tool Validation

| Model | Method | Setup | Validated epochs | Base | Best epoch | Best | Best gain | Last validated | Last gain | Interpretation |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Mistral | GRPO | LoRA r32, lr `1e-6`, 5 epochs | 0-5 | 0.2056 | 5 | 0.2345 | +0.0289 | 5: 0.2345 | +0.0289 | Best Mistral validation signal so far, but absolute accuracy remains low. |
| Mistral | TreeMAPPO | LoRA r32, lr `1e-6`, 5 epochs | 0-5 | 0.2144 | 1 | 0.2193 | +0.0049 | 5: 0.1974 | -0.0170 | Tree run degrades by the end. |
| Mistral | GRPO | LoRA r16, lr `1e-6`, 5 epochs | 0-5 | 0.1943 | 5 | 0.2062 | +0.0119 | 5: 0.2062 | +0.0119 | Stable but small gain. |
| Mistral | GRPO | r16 short safe run, 5 minutes/epoch | 0-4 | 0.2068 | 1 | 0.2224 | +0.0156 | 4: 0.0044 | -0.2024 | Collapsed after epoch 1; not usable as positive evidence. |
| Mistral | GRPO | chunked/resume run, LoRA r32, lr `1e-6` | 0, 3 | 0.1857 | 0 | 0.1857 | +0.0000 | 3: 0.1770 | -0.0087 | Current chunked long run not yet favorable. |

### Mistral No-Tool Serious Test

| Model / checkpoint | DeepMath | MATH | NuminaMath | AMC2023 | CollegeMath | Gaokao 2024 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral base, epoch 0 | 0.2140 | 0.1470 | 0.1900 | 0.0250 | 0.1956 | 0.1154 | 0.1478 |
| Mistral GRPO r32 lr `1e-6`, epoch 5 | 0.2330 | 0.1640 | 0.2500 | 0.0250 | 0.2289 | 0.2308 | 0.1886 |

Interpretation: Mistral GRPO improves over the Mistral base serious-test mean, but the absolute performance is far below Qwen-family results. This is useful as a robustness datapoint, not a headline result.

### Llama No-Tool Status

| Model | Method | Status | Current implication |
|---|---|---|---|
| Llama 3.1 | GRPO | Training reached epoch 16; resume chain is queued. No completed validation/test summary yet. | Cannot report accuracy yet. |
| Llama 3.1 | TreeMAPPO | Rollout reached 22/30 chunks; resume still needed before judging/generation/training. No completed validation/test summary yet. | Cannot report accuracy yet. |

## Negative / Unstable Results

| Model | Setting | Variant | Outcome | Interpretation |
|---|---|---|---|---|
| Qwen2.5 | No-tool | TreeMAPPO rank 32, learning rate `5e-5` | Validation collapsed; epoch 1 was 0.0000 and final epoch 10 was 0.2122. | High LR is unsafe for rank-32 TreeMAPPO. |
| Gemma | No-tool | TreeMAPPO r32, lr `1e-6` | Older 5-epoch run ended below base; chunked run was also below base at epoch 3. | Needs lower LR/rank before concluding model incompatibility. |
| Mistral | No-tool | TreeMAPPO r32, lr `1e-6` | Best epoch was only +0.0049 and final epoch was -0.0170 below base. | Tree setup is not currently stable/effective on Mistral. |
| Mistral | No-tool | GRPO rank 16, earlier safe run | Epoch 1 improved slightly, then accuracy collapsed near zero in later epochs. | Mistral is not paper-primary until stability is understood. |
| Qwen2.5 | No-tool | SGD/no-warmup | Prior partial results showed no clear increasing trend and Tree SGD below base. | Adam and warmup remain standard. |
| Qwen2.5 | No-tool | KL beta `0.04` | Worse than non-KL in prior comparisons. | If KL is used, small beta such as `0.001` is more plausible, but still not clearly helpful. |

## Active / Incomplete Runs

These runs were active or recovering from timeouts as of the latest poll on 2026-08-15.

| Model | Setting | Method | Current state | Use in paper |
|---|---|---|---|---|
| Gemma | No-tool | GRPO | Training reached epoch 28 and validation rollout is running/resubmitted. | Do not cite until validation and serious test complete. |
| Gemma | No-tool | TreeMAPPO | Training reached epoch 26 and resume chain is queued. | Do not cite until validation and serious test complete. |
| Mistral | No-tool | GRPO | Training reached epoch 28 and validation rollout is running/resubmitted. | Secondary robustness only. |
| Mistral | No-tool | TreeMAPPO | Rollout complete; judging repeatedly timed out; 16h cached retry queued. | Blocked on judging completion. |
| Llama 3.1 | No-tool | GRPO | Training reached epoch 16 and resume chain is queued. | Do not cite yet. |
| Llama 3.1 | No-tool | TreeMAPPO | Rollout reached 22/30 chunks; resume still needed. | Do not cite yet. |

## Paper-Ready Claims Supported Right Now

1. The one-shot training pipeline can produce modest but measurable validation gains over base models across several Qwen2.5 and Qwen34 settings.
2. TreeMAPPO is competitive with GRPO in Qwen2.5 no-tool validation and can outperform GRPO in some tool and ablation settings, but current serious tests do not yet prove a uniform aggregate advantage.
3. TreeMAPPO-style credit assignment is sensitive to optimizer and learning-rate choices; rank-32 LoRA with `5e-5` can collapse, while `1e-6` is much more stable.
4. The method-comparison story should be framed carefully: strongest current evidence is “competitive and sometimes better under matched compute,” not “dominates GRPO.”
5. Broader model robustness is still being established for Gemma, Mistral, and Llama due to timeout/recovery work.
