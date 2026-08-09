# Experiment and Scheduling Plan

Last updated: 2026-07-24

## Purpose

This document defines the near-term experimental plan for the TreeMAPPO paper under the current practical constraint:

- We are **not** yet ready to spend production compute on the full `bin_orchestrator.rs` pipeline.
- We are still in the **hyperparameter stabilization / systems validation** phase.
- Therefore, the immediate workflow should prefer the split pipeline:
  1. `oneshot_rollout`
  2. `oneshot_generation`
  3. `oneshot_training`
  4. `oneshot_validation`

This is cheaper, easier to debug, and sufficient for identifying stable training settings before launching multi-epoch orchestrated experiments.

## Scientific Targets

The paper and methodology imply three result families:

1. **Main comparison**
   - Base model
   - GRPO baseline
   - TreeMAPPO

2. **Ablations**
   - TEMPO-style branching ablation
   - TreeRPO-style credit-assignment ablation

3. **Scaling studies**
   - Tool vs no-tool
   - Leaves / branch-budget sweep
   - Model-family sweep

The intended final paper tables are already clear:

- Main table: base vs GRPO vs TreeMAPPO across datasets and model-condition groups.
- Ablation table: TreeMAPPO vs TEMPO-style branching vs TreeRPO-style credit.
- One line chart: performance vs number of tree leaves.

## Current Constraints and Operating Assumptions

### Compute constraints

- Delta GPU training is expensive and queue-limited.
- Single-GPU jobs are materially easier to queue than 4-GPU jobs on Delta.
- CPU-only generation is affordable and should be used aggressively.
- During hyperparameter search, prioritize LoRA single-GPU training. Full-parameter FSDP training is stashed as a backup plan and should not occupy active queue slots unless LoRA results become scientifically unusable or a specific full-model question must be answered.
- Future split-pipeline experiments should use `inference_backend = "vllm"` by default; do not include backend names such as `vllm` in experiment nicknames unless backend behavior itself is the experimental variable.
- Ordinary vLLM experiment jobs should not set a default `VLLM_GPU_MEMORY_UTILIZATION`; let vLLM use its default memory policy unless a specific run explicitly needs a cap.
- Full orchestrator runs are deferred until:
  - training no longer fails due to infrastructure or OOM instability,
  - generation policy is settled,
  - a credible training length cutoff is established for each training regime.
- If time and queue availability permit, run a small number of `bin_orchestrator.rs`
  checks on the basic Qwen2.5 no-tool cases (`grpo_notool` and `tree_notool`).
  These runs are confirmatory checks for the standard online/interleaved setting,
  not the primary paper path. The main experimental focus remains the split
  one-shot workflow because it is cheaper, more debuggable, and better matched
  to the current compute budget.

### Current engineering conclusions

These are operational conclusions, not yet paper conclusions:

1. **The earlier repeated training failures were mostly infrastructure issues, not scientific failures.**
   - A SLURM script bug caused jobs to request the default tiny memory allocation.
   - This invalidates earlier apparent “training OOM” outcomes as evidence about model hyperparameters.

2. **The rollout / generation / training split is the correct near-term workflow.**
   - Rollout artifacts can be reused.
   - Generation is CPU-only and cheap.
   - Training can now be iterated independently.

3. **Distributed training must fail fast on OOM.**
   - For `ddp` / `fsdp`, we should not wrap around the dataset after OOM because ranks may desynchronize.
   - The code now stops on first distributed OOM while still preserving the normal summary/model-save path when possible.

4. **Generation should use by-question grouping for current LoRA experiments.**
   - Current no-tool GRPO generation/training configs use `training_set_sort_mode = "ByQuestion"` so positive/negative samples from the same question are adjacent.
   - Because by-question grouping no longer gives an ascending length prefix, single-GPU LoRA training now uses a synthetic OOM preflight to estimate a safe trajectory cutoff before real optimization.

## Finished Experiments

This section includes both scientific experiments and completed infrastructure-validation runs that materially change the plan.

### A. Training-set generation for no-tool GRPO completed successfully

- Job: `20216392`
- Config family:
  - rollout nickname: `grpo_notool_rollout`
  - generation nickname: `grpo_notool_generation`
- Status: completed

Conclusion:

- The reusable no-tool GRPO rollout artifacts are valid.
- CPU-only generation is viable on Delta.
- We can reuse this generated dataset for both LoRA and full-parameter training because rollout/generation do not meaningfully depend on LoRA vs non-LoRA.

### B. Serious LoRA no-tool GRPO training completed successfully

- Job: `20217150`
- Training config: `config/oneshot_train/qwen25_train_grpo_notool_lora.toml`
- Status: completed

Observed outcome:

- 3 oneshot epochs completed.
- Validation accuracy:
  - epoch 0 baseline: `0.6636`
  - epoch 1: `0.6793`
  - epoch 2: `0.6813`
  - epoch 3: `0.6820`
- DeepMath:
  - baseline: `0.5631`
  - epoch 3: `0.6178`
- Math:
  - baseline: `0.7822`
  - epoch 3: `0.7758`

Training summaries:

- epoch 1:
  - samples trained this run: `3405`
  - longest non-OOM trajectory length: `644`
- epoch 2:
  - samples trained this run: `3899`
  - longest non-OOM trajectory length: `799`
- epoch 3:
  - samples trained this run: `3600`
  - longest non-OOM trajectory length: `916`

Conclusion:

- The split no-tool GRPO -> generation -> LoRA training pipeline is **scientifically and operationally viable**.
- Training improves average validation accuracy over baseline.
- The strongest observed gain is on DeepMath.
- Improvement on Math is not monotonic, so we should not assume all gains generalize uniformly.
- The run also provides a realistic safe length band: trajectories up to roughly `644-916` tokens were observed without OOM in this LoRA setup.

### C. Multiple earlier “OOM” and crash runs are invalid as scientific evidence

Representative jobs:

- `20215509`
- `20216454`
- `20216723`
- `20217580` (later canceled)

Conclusion:

- These runs are mainly useful as debugging evidence.
- They should **not** be counted as negative scientific results because they were confounded by resource misallocation and incomplete OOM handling policy.

### D. SLURM submission bug fixed

Conclusion:

- Training GPU count now comes from the config-driven submit wrapper.
- `oneshot_training.slurm` no longer hardcodes `1` GPU.
- This unblocks valid FSDP smoke testing.

### E. Distributed OOM policy fixed

Conclusion:

- In `ddp` / `fsdp`, training now fails fast on OOM instead of wrapping around the dataset.
- This gives us a trustworthy `longest_non_oom_trajectory_length` signal for setting the next trajectory cutoff.

### F. Mistral rank-16 by-question partial diagnostic

- Training job: `20280124`
- Validation job: `20280440`
- Config: `config/oneshot_train/mistral_train_grpo_notool_lora_r16_5m.toml`
- Status:
  - training completed normally but stopped after epoch 1 due to CUDA OOM,
  - validation timed out after validating base epoch 0 and trained epoch 1.

Observed outcome:

- Training:
  - samples available: `11238`
  - samples trained in epoch 1: `907`
  - longest non-OOM trajectory length: `2642`
  - stopped due to OOM: `true`
- Validation:
  - epoch 0 base accuracy: avg `0.208535`, DeepMath `0.251208`, Math `0.181159`, NuminaMath `0.193237`
  - epoch 1 partial-training accuracy: avg `0.217969`, DeepMath `0.299766`, Math `0.163934`, NuminaMath `0.190141`

Conclusion:

- The partial epoch-1 Mistral result is not a serious scientific result because only `907` samples trained and validation timed out.
- It is still a useful diagnostic: rank 16 did not immediately show the catastrophic post-training collapse seen in earlier Mistral rank-32 runs.
- The safe rerun should use a lower cutoff around the observed safe band instead of relying only on the original `4096` cap.
- Validation needs more than `00:30:00` if it is expected to cover all epochs; the 30-minute partial validation budget is only sufficient for base plus approximately one trained epoch.

### G. VERL one-step-off batch-size issue diagnosed

- Failed job: `20279914`
- Replacement job: `20285925`
- Config: `config/verl_grpo/qwen25_hybrid_lora_r32_4gpu_3train_one_step_off_smoke.env`
- Status:
  - `20279914` failed before training due to a VERL configuration assertion,
  - `20285925` is queued with the corrected batch settings.

Conclusion:

- The issue was not an OOM.
- In one-step-off separate-async mode with `3` FSDP training GPUs and `1` rollout GPU, VERL validates the actor training world as `n_gpus=3`.
- Therefore `TRAIN_BATCH_SIZE` must be divisible by `3`.
- VERL separate-async also requires `TRAIN_BATCH_SIZE == parameter_sync_step * PPO_MINI_BATCH_SIZE`; with `parameter_sync_step=1`, both values are now set to `12`.

## Running / Queued Experiments

### 1. Mistral no-tool LoRA by-question training

- Completed diagnostic job: `20280124`, `training_mistral_by_question_r16_5m_hdd`
- Partial validation job: `20280440`, `validation_mistral_r16_partial2`, timed out after epoch 1 validation
- Safe rerun job: `20280442`, `training_mistral_r16_safe`, completed successfully in `00:52:07`
- Safe validation job: `20287205`, `validation_mistral_r16_safe`, pending `Priority`
- Safe rerun config: `config/oneshot_train/mistral_train_grpo_notool_lora_r16_5m_safe.toml`
- Training data: `/work/hdd/bhph/zluo8/credit_assignment/results/medium_files/mistral/grpo_notool_generation_1h/training_trajectories/trajectories.msgpack`
- Safe rerun resources: `gpuA100x4`, `bfsl-delta-gpu`, `1 x A100`, `32` CPUs, `32G` memory, walltime `01:39:00`
- Safe validation resources: `gpuA100x4`, `bfsl-delta-gpu`, `1 x A100`, `32` CPUs, `32G` memory, walltime `02:00:00`
- Safe rerun cutoff: `training_trajectory_len_cutoff = 2500`

Purpose:

- Test whether the Mistral collapse observed in the earlier rank-32 setting was rank-related, data-order-related, or due to the previous adapter save/load path.
- Use by-question ordering and shorter per-epoch training to get a faster diagnostic signal.

Decision rule:

- If the safe rerun still OOMs, reduce the cutoff below `2500` or make the synthetic preflight cap mandatory before real training.
- If the safe rerun completes multiple epochs, validate with enough walltime and compare trend against the partial epoch-1 signal.
- If validation collapses, inspect LoRA adapter save/load and tokenizer-level trajectory content before launching more Mistral jobs.
- If validation is stable or improves, promote Mistral rank 16 to a longer LoRA run.

### 2. Qwen34 no-tool LoRA by-question pipeline

- Rollout job: `20280026`, `rollout_qwen34_by_question_6h_hdd`, running on `gpua072`
- Generation job: `20280027`, `generation_qwen34_by_question_6h_hdd`, pending dependency on `20280026`
- Training job: `20280028`, `training_qwen34_by_question_r32_6h_hdd`, pending dependency on `20280027`
- Storage: `/work/hdd/bhph/zluo8/credit_assignment/results`
- Sort policy: `ByQuestion`

Latest rollout progress:

- At elapsed `03:50:23` of `07:09:00`, rollout had passed time segment `5/8`.
- Most recent milestone: elapsed `13500s/21600s`, `9673` finished trees and `64500` finished branches.
- No error has appeared in `.err`; the job is progressing normally.

Purpose:

- Test whether a larger model plus longer rollout/training budget produces a clearer LoRA improvement signal than Qwen2.5-7B and Mistral.
- Reuse the split rollout -> generation -> training workflow without relying on the full orchestrator.

Decision rule:

- If rollout completes, verify action-log size/throughput before generation starts.
- If generation completes, inspect training-set question grouping and length distribution before trusting the training result.
- If training collapses similarly to Mistral, prioritize adapter save/load debugging over more model-family sweeps.

### 3. VERL isolated GRPO check

- Failed job: `20279914`, `verl_one_step_off_4gpu_3train`
- Replacement job: `20285925`, `verl_one_step_off_4gpu_3train_b12`, pending `Priority`
- Config: `config/verl_grpo/qwen25_hybrid_lora_r32_4gpu_3train_one_step_off_smoke.env`
- Resources: `gpuA100x4`, `bfsl-delta-gpu`, `4 x A100`, `32` CPUs, `128G` memory, walltime `00:45:00`
- Batch settings: `TRAIN_BATCH_SIZE=12`, `PPO_MINI_BATCH_SIZE=12`, `TRAINING_GPUS=3`, `ROLLOUT_GPUS=1`

Purpose:

- Provide an isolated GRPO implementation check that bypasses most current infrastructure.
- Use one-step-off / separated training-rollout resources to test whether the basic GRPO method can improve under comparable time constraints.

Decision rule:

- If it fails with memory pressure again, inspect whether training and vLLM worker placement are actually separated before spending more GPU time.
- If it fails with another configuration assertion, treat VERL as still in integration/debug mode rather than a scientific result.
- If it runs, use periodic accuracy reports as an external sanity check against the in-house LoRA pipeline.

### 4. Qwen2.5 no-tool LoRA rank-48 low-learning-rate probe

- Training job: `20287242`, `training_qwen25_r48_lr1e6`, pending `Priority`
- Config: `config/oneshot_train/qwen25_train_grpo_notool_lora_r48_lr1e6_8ep.toml`
- Resources: `gpuA100x4`, `bfsl-delta-gpu`, `1 x A100`, `32` CPUs, `32G` memory, walltime `03:00:00`
- Generation prerequisite: `grpo_notool_generation_by_question` from `/work/nvme/bhph/zluo8/credit_assignment/results`
- Output storage: `/work/hdd/bhph/zluo8/credit_assignment/results`
- Training recipe: LoRA rank `48`, LoRA alpha `96`, learning rate `1e-6`, `8` epochs, `900s` per epoch, nominal cutoff `4096`
- Sort policy: `ByQuestion`
- Cutoff handling: single-GPU LoRA synthetic OOM preflight should run at the start of a fresh job and select `95%` of the longest valid synthetic trajectory length.

Purpose:

- Test whether the weak or noisy LoRA gains are partly due to an overly aggressive learning rate rather than adapter rank alone.
- Keep rank above the earlier rank-32 setting while reducing the update magnitude by `50x`.

Decision rule:

- If validation is stable but flat, keep learning rate search active and compare against rank-16/rank-32 anchors.
- If validation collapses despite `1e-6`, deprioritize larger ranks and inspect data/advantage distribution before adding more capacity.
- If validation improves, use rank-48 low learning rate as the next GRPO no-tool anchor and then test the TreeMAPPO no-tool counterpart.

## Stashed Full-Model Plan

Full-parameter FSDP remains a backup path, not an active focus.

Reactivate only if:

- LoRA improvements remain indistinguishable from noise after the by-question and rank-sweep fixes.
- Adapter save/load is proven correct but LoRA capacity appears insufficient.
- A reviewer-facing comparison explicitly requires full-parameter tuning.

When reactivated:

- Reserve at most one full-model queue slot.
- Use a smoke-sized 4-GPU FSDP job first.
- Fail fast on OOM and use `longest_non_oom_trajectory_length` to set the cutoff.
- Do not queue serious full-model runs until the smoke run has validated the cutoff and memory profile.

## Near-Term Experimental Strategy

We should use a staged adaptive plan instead of launching the full paper sweep immediately.

### Phase 0: Stabilize the LoRA training recipe

Goal:

- Identify one robust no-tool GRPO training recipe for single-GPU LoRA.

Promotion criteria:

- At least one successful serious LoRA run.
- No resource-allocation bug.
- No OOM wraparound in active LoRA by-question runs.
- A known usable synthetic-preflight trajectory cutoff for each model/config.

This phase is the current priority. Full-model FSDP is explicitly stashed while LoRA behavior, ranking, sorting, and adapter persistence are being debugged.

Operational default for future serious runs:

- Use `num_oneshot_epochs = 10` so model selection can be done across epochs.
- Use `oneshot_per_epoch_training_time = 900` seconds and `validation_rollout_secs = 600` seconds by default for serious runs.
- Use `total_time_limit_hours = 4.0` as the default SLURM hard limit for those 10-epoch runs.
- Keep enough scheduler headroom beyond the nominal training/validation budget for model load, startup, checkpoint/model export, and wrapper overhead.

### Phase 1: Lock the no-tool Qwen2.5-7B baseline pair

Goal:

- Produce the first scientifically useful controlled pair:
  - GRPO no-tool
  - TreeMAPPO no-tool

Rationale:

- This is the cheapest and cleanest first paper-quality comparison.
- It avoids tool-use complexity while still testing the central credit-assignment claim.

Promotion criteria:

- Stable GRPO no-tool training
- Stable TreeMAPPO no-tool training
- Base-model evaluation already available or easy to compute
- Optional confirmatory orchestrator runs on `grpo_notool` and `tree_notool`
  if time remains after the split one-shot comparison is stable.

### Phase 2: Expand to tool / ablations / branch-budget

Only after Phase 1 is stable.

Goal:

- Add:
  - tool-enabled Qwen2.5-7B
  - TEMPO-style branching ablation
  - TreeRPO-style credit ablation
  - leaves sweep (`8`, `16`, `32`)

### Phase 3: Cross-model sweep

Goal:

- Qwen3-4B tool / no-tool
- Gemma3, Llama3.1, Mistral7B no-tool

This should happen only after the Qwen2.5 schedule produces a trustworthy main comparison and stable operational defaults.

## Adaptive Decision Policy

The experiment queue should react to new results immediately.

### If LoRA remains unstable or collapses

Then prioritize debugging before broadening:

1. Compare adapter-only vs merged-full save/load on identical one-epoch settings.
2. Inspect tokenized training trajectories for repeated, malformed, or mislabeled samples.
3. Verify validation loads the intended LoRA adapter and base model.
4. Run a shorter rank/control sweep only after persistence is verified.

Do not:

- Launch more model-family sweeps if multiple models collapse in the same way.
- Reactivate full-model FSDP until the LoRA failure mode is understood.

### If LoRA becomes stable

Then prioritize:

1. GRPO no-tool LoRA
2. TreeMAPPO no-tool LoRA
3. LoRA rank sweep for the best no-tool condition
4. Tool-enabled LoRA variants
5. Full-parameter FSDP backup smoke only if needed

Rationale:

- Single-GPU LoRA jobs are much easier to queue and therefore improve iteration speed.
- LoRA already has one successful serious run, so it is the shortest path to paper-usable comparisons.
- Full-parameter runs remain valuable, but should be treated as follow-up confirmation rather than the default exploration path.

### If TreeMAPPO underperforms GRPO in the first no-tool comparison

Do not immediately broaden to more models.

Instead run:

1. leaves sweep
2. TEMPO-style branching ablation
3. TreeRPO-style credit ablation
4. possibly one temperature or branch-budget adjustment

Rationale:

- If the core claim is weak on the first controlled pair, we need diagnosis before scale-out.

### If TreeMAPPO clearly beats GRPO on no-tool Qwen2.5

Then immediately promote:

1. tool-enabled Qwen2.5 comparison
2. Qwen3-4B no-tool comparison
3. ablations

This is the best path to a paper-ready story.

## Preferred Queue Shape: Keep 3-4 Jobs Active

The queue should mix cheap preparatory jobs with one or two expensive training jobs.

### Recommended steady-state mix

1. **One GPU LoRA training job**
   - the default highest-priority serious run or smoke run

2. **One secondary GPU or CPU prep job**
   - only if we are confident it will not be invalidated by the first result
   - examples:
     - LoRA rank sweep run while another LoRA run is pending
     - TreeMAPPO generation prep while a LoRA run is running

3. **One CPU generation / rollout-prep job**
   - prepare the next dataset variant

4. **One CPU analysis / evaluation / artifact extraction job**
   - metrics aggregation
   - summary extraction
   - evaluation-only execution

### What not to queue together

- Multiple expensive full-FSDP runs before we know the correct cutoff
- Tool and no-tool branches of the same method before the no-tool branch is stable
- Large ablation batches before the main GRPO vs TreeMAPPO comparison is validated

## Concrete Next Queue

This is the recommended next queue after the current state.

### Slot 1: Active Mistral LoRA validation

- training job `20280442` completed through 10 epochs.
- validation job `20287205` is pending; when it runs, compare epoch-wise accuracy against the partial epoch-1 result and watch for collapse.

### Slot 2: Active Qwen34 LoRA pipeline

Action:

- let rollout `20280026`, generation `20280027`, and training `20280028` proceed if queue allocation starts.
- inspect each phase boundary before interpreting the final result.

### Slot 3: VERL isolated smoke retry

Action:

- let job `20285925` run with `TRAIN_BATCH_SIZE=12` and `PPO_MINI_BATCH_SIZE=12`.
- if it fails before training again, keep VERL as an integration task and do not let it block the in-house LoRA pipeline.

### Slot 4: Qwen2.5 LoRA rank/control follow-ups

Action:

- run the submitted rank-48 low-learning-rate probe `20287242`.
- keep additional single-GPU LoRA experiments contingent on Mistral validation and Qwen2.5 rank-48 trend.
- the preferred branch remains a LoRA rank and learning-rate sweep around the no-tool GRPO setup.

Suggested initial rank sweep:

- rank 8
- rank 16
- rank 32
- rank 48 with lower learning rate

Concrete config files:

- `config/oneshot_train/qwen25_train_grpo_notool_lora_r8.toml`
- `config/oneshot_train/qwen25_train_grpo_notool_lora.toml` (rank 16 baseline)
- `config/oneshot_train/qwen25_train_grpo_notool_lora_r32.toml`

Reason:

- this is the fastest way to improve the training recipe while preserving queue throughput.

### Slot 5: Prepare TreeMAPPO no-tool generation artifacts on CPU

Action:

- launch / verify the no-tool TreeMAPPO generation pipeline using the already valid non-LoRA rollout prerequisites.

Reason:

- This keeps the next scientifically useful comparison ready while the FSDP smoke waits in queue.

### Slot 6: Prepare evaluation / summary scripts for the first main table row

Action:

- standardize extraction of:
  - baseline accuracy
  - GRPO no-tool accuracy
  - TreeMAPPO no-tool accuracy
  - per-dataset breakdown

Reason:

- Results without immediate aggregation slow down decision-making.

### Slot 7: Full-model path remains stashed

Branch:

- Do not queue more full-model FSDP work while LoRA diagnostics are unresolved.
- Reactivate only if LoRA remains indistinguishable from noise after safe cutoff, by-question grouping, and adapter persistence are validated.

### Recommended LoRA rank sweep submission order

Run these in this order:

1. `config/oneshot_train/qwen25_train_grpo_notool_lora_r8.toml`
2. `config/oneshot_train/qwen25_train_grpo_notool_lora.toml`
3. `config/oneshot_train/qwen25_train_grpo_notool_lora_r32.toml`
4. `config/oneshot_train/qwen25_train_grpo_notool_lora_r48_lr1e6_8ep.toml`

Rationale:

- Rank 8 tests whether we can keep most of the gain with a smaller adapter and lower memory pressure.
- Rank 16 is the known-good anchor and should remain in the sweep for direct comparison.
- Rank 32 tests whether extra adapter capacity buys measurable validation gain.
- Rank 48 with `1e-6` learning rate tests whether larger adapters need substantially smaller updates to avoid collapse or noise.

Decision rule after the sweep:

- If rank 8 is within noise of rank 16, prefer rank 8 for future scale-out.
- If rank 32 clearly improves over rank 16, use rank 32 for the first TreeMAPPO LoRA comparison.
- If rank 16 remains best, keep it as the default LoRA setting.

## Proposed Paper-Oriented Experiment Order

This is the recommended order of scientific value, not submission order.

### Tier 1: must-have before broadening

1. Qwen2.5-7B no-tool base
2. Qwen2.5-7B no-tool GRPO (LoRA-first)
3. Qwen2.5-7B no-tool TreeMAPPO (LoRA-first)
4. Qwen2.5-7B no-tool LoRA rank sweep

### Tier 2: strengthen the main claim

5. Qwen2.5-7B tool GRPO
6. Qwen2.5-7B tool TreeMAPPO
7. Qwen2.5-7B no-tool TEMPO-style branching ablation
8. Qwen2.5-7B no-tool TreeRPO-style credit ablation
9. Qwen2.5-7B leaves sweep (`8`, `16`, `32`)

### Tier 3: generalization across model families

10. Qwen3-4B no-tool GRPO
11. Qwen3-4B no-tool TreeMAPPO
12. Qwen3-4B tool GRPO
13. Qwen3-4B tool TreeMAPPO
14. Gemma3 no-tool GRPO vs TreeMAPPO
15. Llama3.1 no-tool GRPO vs TreeMAPPO
16. Mistral7B no-tool GRPO vs TreeMAPPO

## What Counts as a Finished Scientific Experiment

An experiment should be marked finished only if all of the following are true:

1. The job used valid SLURM resources.
2. The run produced normal summary artifacts.
3. The result is not confounded by an infrastructure bug already known to invalidate interpretation.
4. The final model or evaluation output exists.
5. The result has been added to the experiment tracker for comparison against prior runs.

This rule is important because several earlier jobs were operationally useful but scientifically invalid.

## Immediate Conclusions

As of the latest Delta poll, the most important scientific and scheduling conclusions are:

1. **Qwen2.5 no-tool GRPO LoRA remains the most stable positive baseline.**
   - The matched GRPO r32 `lr=5e-5` run improved from roughly `0.6503` to `0.6703`.
   - The GRPO r48 `lr=1e-6` probe has partial validation through epoch 2: base `0.6729`, epoch 1 `0.6717`, epoch 2 `0.6747`; this is stable but only a very small gain so far.

2. **Qwen2.5 TreeMAPPO is rank- and learning-rate-sensitive.**
   - Tree r16 `lr=5e-5` did not collapse in the earlier validated epochs: base `0.6631`, epoch 1 `0.6861`, epoch 2 `0.6802`.
   - Tree r32 `lr=5e-5` collapsed badly: base `0.6580`, best trained epoch `0.2823`, final epoch `0.2122`.
   - Tree r32 `lr=1e-6` has completed 10 epochs of training without OOM; validation is pending.
   - Tree r16 5-epoch rerun has completed training, with validation pending.

3. **The 75% synthetic OOM preflight fix is operationally effective.**
   - Qwen2.5 tree r32 `lr=1e-6` completed 10 epochs without OOM using a cutoff around `2316`.
   - Qwen2.5 tree r16 5-epoch completed all 5 requested epochs, although epoch 5 still OOM-stopped after `1873` samples.
   - Qwen34 r32 completed 24 epochs without OOM under the fixed per-epoch preflight.

4. **Validation is now the main operational bottleneck.**
   - vLLM validation works for many epochs, but `/update_model` can still time out after backend restart.
   - The r48 rank issue was fixed by rounding vLLM `--max-lora-rank` to supported values, but the retry still timed out when updating to epoch 3.

5. **Mistral is not currently a productive paper path.**
   - Mistral r16 validation showed epoch 1 slightly above base but then collapsed near zero by later epochs.
   - It should be deprioritized unless needed for a negative/generalization appendix.

6. **Full-model FSDP remains stashed.**
   - One-GPU LoRA is still the faster route for actionable evidence.
   - Full-model FSDP should remain a backup plan only if LoRA fails to produce a defensible comparison.

## Current Blockers

As of the latest Delta poll, the project is blocked primarily on pending validation jobs and one recurring validation reliability issue.

1. **Qwen2.5 TreeMAPPO r32 low-learning-rate validation is pending.**
   - Training job `20369711`, `training_qwen25_tree_r32_lr1e6_10ep`, completed in `02:32:53`.
   - Validation job `20369712`, `validation_qwen25_tree_r32_lr1e6_10ep`, is pending with no dependency blocker.
   - This is the key test of whether the r32 collapse was caused mainly by `lr=5e-5`.

2. **Qwen2.5 TreeMAPPO r16 5-epoch validation is pending.**
   - Training job `20370144`, `training_qwen25_tree_r16_5ep_cut075`, completed in `01:09:57`.
   - Validation job `20370145`, `validation_qwen25_tree_r16_5ep_cut075`, is pending with dependency satisfied.
   - This is the key test of whether the earlier r16 improvement persists beyond the first two validated epochs.

3. **Qwen34 validation needs a retry to move past base epoch.**
   - Training job `20358896`, `training_qwen34_r32_24ep_cut075_epochfix`, completed all 24 epochs.
   - Initial validation job `20358897` validated only base epoch: avg `0.7055`, then failed on `/update_model` timeout for epoch 1.
   - Retry job `20377846`, `validation_qwen34_r32_24ep_retry`, is pending.

4. **Qwen2.5 GRPO r48 validation is partial.**
   - Training job `20362732`, `training_qwen25_r48_lr1e6_3ep_cut075`, completed all 3 epochs.
   - Rank-fix validation job `20377845` validated epochs 1 and 2 after fixing vLLM max-rank handling, but failed updating to epoch 3.
   - Current partial result: base `0.6729`, epoch 1 `0.6717`, epoch 2 `0.6747`; this is stable but not a meaningful gain yet.

5. **vLLM model-update timeout is recurring.**
   - Failures appear when switching LoRA adapter epochs via `/update_model`; the backend exits and relaunch sometimes does not become healthy within 300 seconds.
   - Obvious config bug for rank 48 was fixed by rounding `--max-lora-rank` up to the nearest supported value, but timeout remains possible.
   - If this continues, the next infrastructure fix should be to validate each epoch in a fresh process/job or make validation restart the wrapper per epoch instead of relying on update-in-place.

## Macroscopic Setbacks

These are project-level setbacks that explain why the schedule remains adaptive and why the current priority is LoRA stabilization rather than broad experiment scaling.

1. **Accuracy gains remain modest and recipe-dependent.**
   - GRPO r32 has a small positive signal around `+0.020` average accuracy.
   - Tree r16 has a stronger early signal around `+0.023`, but needs the pending 5-epoch validation for confirmation.
   - Tree r32 at `lr=5e-5` collapsed, so TreeMAPPO is not yet robust across LoRA rank.

2. **Higher-rank LoRA requires lower learning rates and careful validation.**
   - Tree r32 `lr=5e-5` collapsed despite successful training.
   - GRPO r48 `lr=1e-6` appears stable through epoch 2 but does not yet show a meaningful gain.
   - Tree r32 `lr=1e-6` is the active test of whether lower LR rescues higher-rank TreeMAPPO.

3. **Validation remains operationally fragile.**
   - vLLM is substantially better than the earlier sglang path, but adapter epoch switching can still fail.
   - This directly affects qwen34 and r48 completion, and may require a fresh-process-per-epoch validation mode.

4. **Mistral results are currently negative.**
   - Mistral r16 safe validation showed base `0.2068`, epoch 1 `0.2224`, then near-zero accuracy by epochs 2--4.
   - This suggests model-family sensitivity or overtraining/collapse; it is not a priority for the main paper evidence.

5. **Full-model FSDP remains too slow for rapid iteration.**
   - Four-GPU full-model jobs queue more slowly and are harder to recover after OOM.
   - They remain a backup route rather than an active blocker.

6. **Storage and artifact pruning continue to affect bookkeeping.**
   - Model checkpoint pruning saves quota but must be launched manually after diagnostics/testing no longer need intermediate checkpoints.
   - Some prerequisites still live on NVMe while newer outputs are on HDD.

7. **The main scientific comparison is close but not finalized.**
   - We have GRPO r32 stable positive evidence and Tree r16 early positive evidence.
   - The pending Tree r16 5-epoch validation and Tree r32 `lr=1e-6` validation should determine whether TreeMAPPO has a defensible no-tool improvement over GRPO or whether the claim must be narrowed.

## Implementation Audit Notes

These notes record known implementation-paper mismatches that should not block the immediate abstract-oriented experiment schedule, but should be resolved or explicitly scoped before a final paper submission.

1. **Guided branching features are now runtime-controlled but not yet rerun.**
   - One-shot rollout supports `enable_uncertainty_aware_branching` and `force_selected_branch_token` via TOML fields or CLI flags.
   - The forced-token path now preserves the selected token's old rollout log-probability instead of inserting it with synthetic logprob `0.0`, which is important under the clipped policy-ratio training objective.
   - Deferred qwen25 no-tool TreeMAPPO configs are prepared for a serious uncertainty-aware + forced-token run: `config/oneshot_rollout/qwen25_rollout_tree_notool_uncert_forced_4h.toml`, `config/oneshot_generation/qwen25_generate_tree_notool_uncert_forced_4h.toml`, and `config/oneshot_train/qwen25_train_tree_notool_uncert_forced_lora_r32_lr1e6_10ep_15m.toml`.
   - Do not submit this run while the queue is heavily contended; submit it later as the next strict test of the full guided-branching mechanism.

3. **GRPO configs now use explicit group-normalized terminal reward credit.**
   - `GrpoTerminalReward` was added as a separate advantage policy in parallel with `TreeRpoWinRate`.
   - The policy asserts a flat rollout shape (`num_trunks == num_leaves`) and assigns group-normalized terminal correctness to each response trajectory's supervised tokens.
   - Existing experiment results were not rerun after this bookkeeping fix; rerunning GRPO generation/training is deferred unless the final paper needs a strictly regenerated baseline.

4. **Training now uses a clipped policy-ratio surrogate instead of direct advantage-weighted CE.**
   - Token advantages are still clipped to `[-3, 3]` during training-set materialization and clamped again by `advantage_clip = 3.0` in Python training.
   - Generated training trajectories now store old rollout token log-probabilities, and Python training uses a fixed PPO/GRPO-style clip ratio of `0.2`.
   - Existing training sets without `old_logprobs` must be regenerated before future training, because the new standard objective intentionally has no old-objective fallback path.

## Latest Completed Test Update — 2026-08-01

All tracked held-out serious test jobs from the latest batch completed, and the Delta queue was empty at the last poll.

- Qwen34 tool GRPO best checkpoint: held-out mean accuracy `0.6609` across six datasets.
- Qwen34 tool TreeMAPPO best checkpoint: held-out mean accuracy `0.6478` across six datasets.
- Qwen2.5 no-tool positive-advantage-only TreeMAPPO: held-out mean accuracy `0.6800`.
- Qwen2.5 tool positive-advantage-only TreeMAPPO: held-out mean accuracy `0.6117`.

Current interpretation:

1. Qwen2.5 no-tool remains the cleanest paper path because both GRPO and TreeMAPPO show modest positive validation signals under matched rank-32, learning-rate `1e-6` settings.
2. Qwen34 tool is a mixed/negative comparison for TreeMAPPO: the Tree checkpoint is below the matched GRPO checkpoint on aggregate held-out accuracy, although it improves on a subset of datasets.
3. Positive-advantage-only TreeMAPPO should be treated as an ablation. It is not currently stronger than the standard TreeMAPPO story, especially in the tool setting.
4. The next scientific decision should be whether to replicate the Qwen2.5 no-tool GRPO-vs-Tree comparison for robustness or spend the next queue slots on planned ablations such as branch budget, TEMPO-style branching, and uncertainty-aware forced-token branching.


## Qwen2.5 No-Tool GRPO-vs-Tree Replication — Submitted 2026-08-01

Purpose: independently replicate the matched Qwen2.5 no-tool comparison under the current preferred LoRA recipe.

Matched setup:

- Model: Qwen2.5-7B no-tool.
- Methods: GRPO terminal-reward baseline vs TreeMAPPO posterior credit assignment.
- LoRA: rank `32`, alpha `64`, dropout `0.05`.
- Learning rate: `1e-6`.
- Training: 5 epochs, 15 minutes per epoch.
- Rollout: independent replicate nicknames with 2 hours of rollout per method.
- Generation sorting: `ByQuestion`.
- Backend: vLLM.
- Storage: `/work/hdd/bhph/zluo8/credit_assignment/results`.

Submitted chain:

- GRPO rollout: `20745568`, `qwen25_rep1_grpo_rollout`, 1 A100, 32 CPUs, 32G, `02:30:00`, account/QOS `bfsl-delta-gpu`, partition `gpuA100x4`.
- Tree rollout: `20745569`, `qwen25_rep1_tree_rollout`, 1 A100, 32 CPUs, 32G, `02:30:00`, account/QOS `bfsl-delta-gpu`, partition `gpuA100x4`.
- GRPO generation: `20745570`, depends on `20745568`, 32 CPUs, 32G, `01:00:00`, account/QOS `bfsl-delta-cpu`, partition `cpu`.
- Tree generation: `20745571`, depends on `20745569`, 32 CPUs, 32G, `01:00:00`, account/QOS `bfsl-delta-cpu`, partition `cpu`.
- GRPO training: `20745572`, depends on `20745570`, 1 A100, 32 CPUs, 32G, `02:00:00`, account/QOS `bfsl-delta-gpu`, partition `gpuA100x4`.
- Tree training: `20745573`, depends on `20745571`, 1 A100, 32 CPUs, 32G, `02:00:00`, account/QOS `bfsl-delta-gpu`, partition `gpuA100x4`.
- GRPO validation: `20745574`, depends on `20745572`, 1 A100, 32 CPUs, 32G, `01:30:00`, account/QOS `bfsl-delta-gpu`, partition `gpuA100x4`.
- Tree validation: `20745575`, depends on `20745573`, 1 A100, 32 CPUs, 32G, `01:30:00`, account/QOS `bfsl-delta-gpu`, partition `gpuA100x4`.

Decision rule: compare validation gains relative to each replicate's base checkpoint, then run held-out serious tests on the best GRPO and Tree checkpoints if both validation jobs complete cleanly.

## Breaking Pipeline Redesign Plan — 2026-08-04

This section records the planned breaking change from walltime-based one-shot experiments to chunk-size-based, crash-resumable experiments. The goal is to make rollout, training-set generation, training, and validation resumable with simple file-level state rather than complicated in-process checkpointing.

### Core Design Goals

1. **Question chunks replace walltime budgets.**
   - One dataset chunk corresponds to one training epoch.
   - Rollout and training should be specified by chunk size / chunk identity rather than by rollout seconds or per-epoch training seconds.
   - Chunk files should be independently reproducible and safe to resume after crashes.

2. **Chunk assignment is deterministic.**
   - `bin_oneshot_rollout` should accept `--num-questions-per-chunk`.
   - Questions are preassigned to chunks in ascending `flat_id` order.
   - Chunk size counts raw candidate questions before knowing whether each question will survive all-correct/all-incorrect filtering.
   - Training rollout chunks should contain the same number of raw questions, except possibly the final remainder chunk.
   - Async rollout completion order must not affect question-to-chunk assignment.
   - Rollout outputs should be split into multiple files/directories, one per question chunk.

3. **Training-set generation follows rollout chunking.**
   - Training trajectories are naturally chunked according to the same deterministic question chunks.
   - Generated trajectory chunks may contain fewer questions than rollout chunks because all-correct/all-incorrect questions are filtered.
   - Do not patch vacancies after filtering; chunk membership remains determined only by the original raw question mapping.
   - Sorting is fixed to `ByQuestion`; remove or deprecate alternative training-set sort choices for the new pipeline.
   - Keep filtering out questions whose outcomes are all correct or all incorrect, because they provide no within-question contrast.
   - Remove filtering of individual trajectories based on low average advantage scale.

4. **Segment advantages are shared across all containing trajectories.**
   - Current generation assigns a segment's advantage to only one selected trajectory, then zeros the segment before selecting the next trajectory.
   - New generation should distribute each segment's advantage uniformly across all materialized trajectories containing that segment.
   - Without trajectory filtering, trajectory count should equal leaf count for each surviving question, so this is equivalent to dividing by all containing leaves.
   - This preserves the same total segment-level training weight while avoiding arbitrary single-trajectory ownership of shared prefixes.

5. **Gradient accumulation is question-aware.**
   - Replace `grad_accum_steps` with `min_grad_accum_steps`.
   - Batch size is fixed to `1`, so `min_grad_accum_steps` counts trajectories / microbatches.
   - Training accumulates gradients until at least `min_grad_accum_steps` trajectories have been included and all trajectories for the most recent question in the accumulation window have also been included.
   - This rounds optimizer steps up to full question groups, preventing trajectories from the same question from being split across different optimizer updates.
   - Do not add a hard maximum accumulation cap, but print a warning if an optimizer step accumulates more than `2 * min_grad_accum_steps` trajectories.

6. **Training-set validation becomes chunk-aware.**
   - In addition to held-out validation, run validation on training chunks before and after training on each chunk.
   - Training-chunk validation should rerun inference on the raw questions in the chunk, not evaluate only materialized training trajectories.
   - Training-chunk validation uses deterministic inference with temperature `0`, not the training rollout temperature.
   - Training-chunk validation after chunk `k` should evaluate the trained model/checkpoint corresponding to epoch/chunk `k`.
   - This should expose whether the model improves on the exact chunk it trained on and whether improvements transfer to held-out validation.
   - The validation record should identify model checkpoint, chunk id, split type (`training_chunk` vs held-out validation), and whether it is pre- or post-training for that chunk.

7. **Held-out validation remains unchanged.**
   - The current 3000-sample validation policy should remain the held-out validation path.
   - Held-out validation runs for every trained epoch/checkpoint.
   - Held-out validation also runs on the base model as epoch `0`.

### Expected Pipeline Shape

1. Rollout chunk `k`: generate deterministic action-log artifact for questions in chunk `k`.
2. Generate training chunk `k`: materialize all eligible question-grouped training trajectories for that rollout chunk.
3. Validate pre-training on training chunk `k`.
4. Train on training chunk `k`, with question-aware gradient accumulation.
5. Save checkpoint for epoch/chunk `k`.
6. Validate post-training on training chunk `k`.
7. Validate held-out set for checkpoint `k`.
8. Continue with chunk `k+1`.

### Implementation Notes

- The chunk id should become part of artifact paths and summaries so partial completion is visible from the filesystem.
- The new pipeline should prefer append-safe or write-then-rename artifact creation to avoid treating partially written chunks as complete.
- Each completed chunk/epoch should write an explicit marker file, such as a separate dummy completion file.
- Detection logic should be simple: if the marker is absent, erase stale artifacts for that chunk/epoch and restart from that whole chunk.
- The recovery unit is the whole chunk; partial chunk resume is intentionally not supported.
- Existing walltime-based configs and launchers will likely need breaking TOML schema changes rather than backward-compatible fallbacks.
- Existing experiment results should be treated as legacy results; do not silently mix old unchunked artifacts with new chunked artifacts.
- The first smoke test should use a small chunk size to verify deterministic chunk mapping, trajectory materialization, question-aware accumulation boundaries, and per-chunk validation.

### Resolved Checkpoint-Pruning Decision

1. **Disable LoRA checkpoint pruning during initial chunked-pipeline smoke tests.**
   - LoRA checkpoint pruning refers to the existing validation-side disk cleanup that can delete non-best LoRA checkpoint artifacts after validation.
   - Disable this cleanup temporarily while validating the new chunk artifact layout, completion-marker logic, and resume semantics.
   - After the chunked pipeline is stable, re-enable pruning with chunk-aware rules.

## Separate Judging-Pass Redesign Plan — 2026-08-04

This section records the planned breaking change to decouple leaf judging from rollout, validation, and testing. The goal is to make correctness judgments deterministic, reusable, and consistent across experiments for the same question/answer input.

### Core Design Goals

1. **Judging becomes a separate pass.**
   - Training rollout should generate trajectories without finalizing correctness judgments inline.
   - The new implementation should not produce inline judgments at all.
   - Training-chunk validation, held-out validation, and testing should also separate model inference from answer judging.
   - Validation and testing should always launch a separate judging job; even exact string matches between model answer and reference answer should be judged exclusively by the judging job.
   - Existing behavior that uses leaf judgment results during branching should be removed for simplicity and better pass decoupling.
   - Rollout branching should rely only on non-judgment state, such as rollout structure, branch budget, and model-token signals.

2. **Judgments are centralized and reusable.**
   - Maintain a centralized judgment cache keyed by the stable judgment input.
   - The judgment input should match the current implementation in `judge_correctness.rs`.
   - The cache key should include split, question `flat_id`, and the model-provided answer string.
   - Reuse cached judgments across training rollout, training-chunk validation, held-out validation, testing, and different experiment variants.
   - Given the same judgment input, all experiments should observe the same verdict unless the cache schema/version explicitly changes.
   - Old judgment caches should be discarded rather than migrated.
   - Existing old embedded verdicts in legacy artifacts should be ignored.
   - Do not invest implementation effort in old unchunked artifacts; the new pipeline should use new chunked rollout files.

3. **Judgment cache files are split-aligned and human-readable.**
   - Use readable JSONL files rather than the current database format.
   - Cache files should be chunked by split and `flat_id`, with each chunk responsible for 1000 raw questions.
   - Different splits correspond to different cache chunk namespaces/files; the chunk file is determined by split and `flat_id`.
   - Cache chunk size is independent of rollout chunk size.
   - Judging logic should locate cache chunks by split and `flat_id`, not by attempting to map rollout chunks onto cache chunks.
   - Linear scan within a 1000-question JSONL chunk is acceptable for inspection and access.
   - Each cache chunk should keep one canonical record per `(split, flat_id, answer_string, cache_version)` key.
   - If a canonical decision already exists for a key, the item should not be judged again under the same cache version.

4. **Judging jobs are parallel but per-chunk serialized.**
   - Judging jobs should run on CPU nodes because judging uses API models rather than local GPU inference.
   - API requests should be asynchronous for throughput.
   - Slurm judging jobs may be queued in parallel, but only one job may acquire the lock for a given cache chunk at a time.
   - Do not differentiate read and write locks for the first implementation; one exclusive per-chunk lock is simpler and safer.
   - Judging job submission should be independent of rollout chunk size and cache chunk size.
   - Each judging job should judge all trajectories produced by the immediately previous inference/rollout pass, which may span multiple rollout chunks and multiple cache chunks.
   - Judging results should be dispatched to the appropriate cache chunks based on split and `flat_id`.
   - Use separate request concurrency of `200` per model to maximize judging throughput.
   - Prefer correctness and race freedom over maximum judging throughput.

5. **OpenRouter is the judging provider.**
   - All judging models should be accessed through OpenRouter.
   - Use the same judging prompt for all models unless a specific model is empirically shown to be incompatible with the prompt.
   - API calls should use up to three retry attempts.
   - If all three attempts are exhausted and no valid judge output is produced, write back current cache/progress state and exit immediately with exit code `1`.

6. **Multi-phase agreement is the default judgment protocol.**
   - Phase 1 uses three independent lightweight judges: `deepseek-v4-flash`, `qwen3-32B`, and `google/gemini-2.5-flash-lite`.
   - If all three phase-1 models agree, that unanimous verdict is final and cached.
   - If phase 1 disagrees, phase 2 uses two stronger independent judges: `deepseek-v4-pro` and `gpt-4.1-mini`.
   - Phase 2 concludes only if `deepseek-v4-pro` and `gpt-4.1-mini` agree.
   - If the two phase-2 judges disagree, phase 3 uses `gpt-5-mini` as the exclusive final-verdict model.
   - Judges should remain independent unless the final verdict has already been determined.
   - Store individual judge outputs only for disagreement/escalation cases; in the worst case, record all six outputs from phases 1, 2, and 3.
   - Escalated final-verdict cases should be recorded both in the original cache chunk and in one append-only global JSONL shared across all experiments, for human notification and later audit.

7. **Judging statistics become first-class outputs.**
   - Report total judged items, cache-hit rate, unanimous-agreement rate, phase-2 rate, final-escalation rate, and expensive-judgment percentage.
   - Record per-dataset and per-experiment statistics where applicable.
   - Keep enough metadata to identify whether high-cost judging is concentrated in specific datasets, models, or experiment variants.

### Expected Pass Structure

1. Inference/rollout pass emits raw model outputs and stable judgment inputs without correctness verdicts.
2. Judging pass reads unresolved judgment inputs, checks the centralized cache, and schedules only cache misses.
3. Phase-1 three-model async judging panel runs on CPU nodes for cache misses.
4. Phase-2 two-model stronger judging runs only for phase-1 disagreements.
5. `gpt-5-mini` final-verdict escalation runs only if disagreement remains after phase 2.
6. Judging pass writes final verdicts, escalation audit records, and judging statistics.
7. Downstream generation, training-set filtering, validation summaries, and test summaries consume cached final verdicts.

### Dependency Structure

- Training rollout judgment should be a separate Slurm job depending on training rollout.
- Training-set generation should be a separate Slurm job depending on training rollout judgment.
- Training-chunk validation, held-out validation, and testing should always use separate judging jobs after inference jobs.

### Implementation Notes

- The judgment cache should use write-then-rename marker semantics similar to the chunked pipeline where applicable.
- The cache schema should include a version so prompt/model/protocol changes do not silently mix old and new judgments.
- Existing experiment results and embedded inline judgments should be treated as legacy and ignored by the new chunked pipeline.
- There are no remaining known design clarifications for the judging-pass plan; implementation may still surface code-level details.

## Removing Inline Tree Judging — Implementation Plan

This section records the concrete code-level plan for removing inline judging from tree construction and reconnecting judgments through a separate judging pass.

### Current Coupling To Remove

1. **Tree action log currently contains judgments.**
   - `DirectTreeAction::JudgeAnswer(CorrectnessJudgment)` runs during rollout.
   - `DirectTreeAction::AttachSegmentToTree { ..., correctness_judgment }` attaches both the generated segment and its verdict.
   - `DirectTree::leaf_segment_judgments` is populated during `AttachSegmentToTree`.

2. **Tree state machine currently has judging statuses.**
   - Trunk, guided branching, and spontaneous branching each have a `JudgingSegment` status.
   - `tree_to_action.rs` calls `judge_final_answer` when the tree reaches `JudgingSegment`.
   - This makes rollout depend on API judging and old judgment-cache behavior.

3. **Branching / early stopping currently can depend on correctness.**
   - Training rollout can early-stop when enough leaves have been judged and the tree is all correct or all incorrect.
   - This must be removed so rollout no longer needs correctness verdicts.

### New Tree Construction Contract

1. **Rollout emits only raw tree structure and final answers.**
   - Keep `SubmitAnswer(FinalAnswer)` as the event that ends a leaf trajectory.
   - Replace judged attachment with an unjudged leaf attachment action, for example `AttachSegmentToTreeUnjudged { parent_segment_id, finalized_content_array, final_answer }`.
   - The attached leaf segment should store or be associated with its `FinalAnswer`, but not `CorrectnessJudgment`.
   - The new chunked rollout action log should never emit `JudgeAnswer` or judged `AttachSegmentToTree`.

2. **Tree reconstruction has two views.**
   - `UnjudgedDirectTree`: reconstructs structure, leaf segment ids, trunk leaves, and final answers, but has no verdicts.
   - `JudgedDirectTree`: combines an unjudged tree with a judgment overlay keyed by leaf segment id or by stable judgment input.
   - Downstream code that calculates posteriors, advantages, accuracy, or all-correct/all-incorrect filtering must require `JudgedDirectTree` or an equivalent explicit judged view.

3. **Leaf identity must be stable within chunk artifacts.**
   - The judging job needs to connect verdicts back to the tree deterministically.
   - Each emitted judgment request should include experiment artifact id, split, flat id, tree id/question id, leaf segment id, final answer, and the exact model answer string used in the cache key.
   - The final cache lookup key remains `(split, flat_id, answer_string, cache_version)`, but the per-artifact judgment overlay should also record the leaf segment id so the tree can be annotated without ambiguity.

### Judging Job Connection To Trees

1. **Request extraction pass.**
   - A judging job reads the previous inference/rollout artifacts and reconstructs unjudged trees.
   - For each leaf with a final answer, it emits a judgment request record matching the current `judge_correctness.rs` input semantics plus split, flat id, and answer string.
   - Failure final answers should still produce a deterministic cached verdict through the judging pass; do not short-circuit in rollout.

2. **Cache resolution pass.**
   - For each request, check the chunked JSONL judgment cache by split and flat id.
   - If a canonical verdict already exists for `(split, flat_id, answer_string, cache_version)`, reuse it.
   - Otherwise run the OpenRouter multi-phase judging protocol and write the canonical verdict to the appropriate cache chunk.

3. **Overlay materialization pass.**
   - After cache resolution, write a per-rollout/per-validation judgment overlay file adjacent to the inference artifact.
   - The overlay should map each tree leaf to a verdict, for example `{ tree_key, split, flat_id, leaf_segment_id, answer_string, cache_key, is_correct }`.
   - Write the overlay through a temporary file and then atomically create a completion marker.
   - Downstream generation/validation/test summary jobs should require the overlay completion marker before reading judgments.

4. **Judged tree loading API.**
   - Add a loader that takes an unjudged tree artifact and its overlay, validates that every leaf has exactly one verdict, and returns a judged view.
   - The loader should fail fast if a leaf is missing, duplicated, or mapped to a cache verdict whose answer string does not match the leaf final answer.
   - This loader becomes the only supported path for training-set generation, posterior calculation, TreeRPO/GRPO advantage calculation, accuracy aggregation, and all-correct/all-incorrect filtering.

### Migration Steps

1. Add new unjudged leaf attachment action and tree fields for leaf final answers.
2. Remove `JudgeAnswer` production from `tree_to_action.rs`.
3. Remove inline `judge_final_answer` calls from rollout.
4. Remove judged early stopping from tree expansion; stop by deterministic trunk/leaf budget only.
5. Add judgment-request extraction from unjudged tree artifacts.
6. Add chunked JSONL judgment cache and OpenRouter judging protocol.
7. Add judgment overlay writer and completion marker.
8. Add judged-tree loader that combines unjudged trees with overlays.
9. Update training-set generation, validation summaries, testing, browsing, and accuracy utilities to consume judged views.
10. Treat all old judged action logs as legacy; do not support mixing old inline-judged artifacts with the new chunked pipeline.

### Code Hotspots

- `src/tree_action.rs`: remove or legacy-gate `JudgeAnswer`; replace judged attach action with unjudged attach action for new artifacts.
- `src/tree_status.rs`: remove `JudgingSegment` and `AttachingToTree` states that carry `CorrectnessJudgment`; keep only states needed to attach raw generated leaves.
- `src/tree_to_action.rs`: remove `judge_final_answer` calls and any rollout-time correctness dependency.
- `src/tree_from_action.rs`: attach generated leaf segments and final answers without populating `leaf_segment_judgments`.
- `src/tree.rs`: split raw leaf final-answer storage from judged overlay storage.
- `src/tree_posterior.rs`, `src/tree_advantage.rs`, `src/training_set.rs`, `src/get_accuracy.rs`: require judged-tree inputs.
- `src/judge_correctness.rs`: keep prompt/input semantics but move execution to a separate judging binary/job using OpenRouter and chunked JSONL cache.

## Direct Tree Artifact Storage Plan

This section records the planned switch from action-log rollout artifacts to finalized tree rollout artifacts for the new chunked pipeline.

### Artifact Policy

1. **Finalized chunks store trees directly.**
   - Rollout chunks should write finalized unjudged trees as the primary production artifact.
   - Action logs take too much space and should no longer be the default downstream dependency for finalized chunks.
   - Keep action-log infrastructure in the codebase as an optional debug/fallback path in case action-level replay is needed later.
   - Do not remove action-log types immediately; legacy/debug code may still use them.

2. **Downstream readers support both sources.**
   - Judging should read direct tree artifacts first and action logs only as an explicit fallback/debug mode.
   - Training trajectory generation should support direct tree input apart from action-log reconstruction.
   - Tree browsing should support direct tree files apart from action-log files.
   - Accuracy and validation summary utilities should consume direct tree + judgment artifacts in the new pipeline.

3. **Tree and judgment data are separate.**
   - Define a tree struct that contains only rollout-produced structure and final answers, with no correctness verdicts.
   - Define a separate tree-judgment struct that contains judgment results.
   - Define `TreeJudged` as an explicit combined struct containing both tree and judgment fields.
   - This makes the pass boundary concrete: rollout produces trees; judging produces judgments; downstream training/validation consumes `TreeJudged`.

4. **Judgment mapping avoids leaf-index fragility.**
   - The tree judgment should include a mapping from model answer string to boolean verdict.
   - This is intended to avoid index-mismatch bugs when connecting judgments back to tree leaves.
   - The judged-tree loader should validate that every model-provided leaf answer in the tree has exactly one verdict in the judgment mapping.
   - If multiple leaves produce the same answer string, they intentionally share the same cached judgment verdict.

### Direct Tree Artifact Contents

Each direct tree artifact should preserve enough information for judging, training, browsing, and debugging:

- Question metadata: split, flat id, dataset/source, raw question, reference answer.
- Rollout metadata: model, config nickname, rollout config, backend, chunk id, artifact schema version.
- Tree structure: segment ids, parent ids, child ids, trunk leaf ids, root id.
- Segment contents: prompt/tool response/supervised model tokens, token ids, decoded text where useful, and rollout old logprobs.
- Leaf outputs: final answer object and normalized model answer string for judgment-cache lookup.
- Optional debug provenance: branch decisions, selected branch token/logprob, seeds if available, and generation parameters.

### Migration Steps

1. Add direct serializable tree artifact structs for unjudged trees.
2. Add `TreeJudgment` and `TreeJudged` structs with explicit pass boundaries.
3. Add writer for finalized chunk tree artifacts and completion markers.
4. Make rollout write direct tree artifacts by default, with action logs optional/debug-only.
5. Add direct tree readers for judging, training-set generation, tree browsing, validation summaries, and testing.
6. Keep action-log replay readers behind explicit fallback/debug flags.
7. Smoke test direct tree round-trip: rollout tree -> judgment overlay -> `TreeJudged` -> generation/browse/accuracy.

### Implementation Status As Of 2026-08-04

Completed:

1. **Direct tree artifacts.**
   - Added direct unjudged tree artifacts, per-chunk tree files, and `chunk_{n}_done` markers.
   - Rollout now writes completed chunk artifacts as soon as all questions in a chunk are complete, while retaining action logs as debug/replay infrastructure.

2. **Separate judging pass infrastructure.**
   - Added chunked JSONL judgment cache and `bin_oneshot_judging`.
   - Added a reusable CPU-only SLURM launcher for judging jobs.
   - Added a dependency-aware one-shot pipeline launcher for rollout -> training-judging -> generation -> training -> validation.
   - Added `TreeJudgment` and `TreeJudged`; generation, validation accuracy, and testing accuracy can now consume direct trees plus judgment overlays.

3. **Training-set generation changes.**
   - Direct judged-tree generation is implemented.
   - The old low-advantage trajectory filter is removed.
   - Segment advantages are now split uniformly across all leaves/trajectories containing the segment.
   - Generated trajectory output is fixed to question ordering for the new pipeline and can additionally emit `chunk_{n}.msgpack` trajectory files by preassigned question flat-id ranges.

4. **Training accumulation changes.**
   - `grad_accum_steps` is now semantically treated as `min_grad_accum_steps` and serialized as `min_grad_accum_steps`; old TOML keys are accepted as an alias.
   - Single-process training rounds accumulation up to include all trajectories from the last question in the accumulation window.
   - Training logs a warning if the rounded accumulation window exceeds twice the configured minimum.
   - Planned follow-up: extend the single-GPU LoRA synthetic preflight to estimate a conservative `max_batch_tokens` budget for adaptive batching. The current preflight only finds a safe single-trajectory length cutoff; it does not yet binary-search a safe padded batch tensor budget. The adaptive-batching preflight should account for the fact that input tensors scale as `batch_size * longest_trajectory_length_in_batch`, then apply a safety multiplier before enabling packed microbatches.

5. **Training checkpoint status resume.**
   - One-shot training now stores epoch-boundary optimizer/status state as `training_status.pt` next to each completed `oneshot_epoch_{n}` model checkpoint.
   - The status includes optimizer state and global optimization step, so resumed chunk-backed training can continue Adam/SGD state and learning-rate schedule from the latest completed epoch.
   - After a new epoch checkpoint and status are written, the previous epoch's `training_status.pt` is removed, keeping only the latest resume status.
   - Legacy model checkpoints without `training_status.pt` remain valid: training resumes from the model weights and initializes optimizer/status fresh.

6. **Chunk-backed training resume.**
   - One-shot training now prefers generated `chunk_{n}.msgpack` trajectory files when present.
   - Each chunk maps to one one-shot epoch, and recovery resumes at the first missing epoch model artifact.
   - If prior one-shot artifacts were created with a different `num_oneshot_epochs`, training no longer exits early; it refreshes the run manifest to the current request and appends from the first missing contiguous epoch when possible.

Current implementation status and remaining verification:

1. **Training-chunk validation.**
   - Implemented as a diagnostic-only separate binary: `bin_oneshot_training_chunk_validation`.
   - It reads the generated `training_trajectories/chunk_{n}.msgpack` file as the source of truth and collects the observed unique training-question flat ids.
   - It reruns deterministic inference on exactly the observed flat ids, not the pre-filter chunk range, so filtered-out questions do not get vacancy-patched back into diagnostics.
   - For `num_oneshot_epochs = N`, diagnostic validation performs `2N` validations: for each chunk/epoch `n`, it evaluates checkpoint `n - 1` before training on chunk `n` and checkpoint `n` after training on the same chunk.
   - It records chunk index, observed trajectory count, observed question count, min/max observed flat id, before/after epoch, before/after accuracy, and artifact paths at `{mount_dir}/small_files/{model}/{config}/training_chunk_validation/chunk_{n}.json`.
   - It supports explicit `--phase rollout`, `--phase judge`, and `--phase score`; `all` remains available as a compatibility/debug path.
   - The pipeline launcher schedules one all-chunk rollout job, one all-chunk judging job, and one all-chunk diagnostic scoring job for the whole experiment, rather than three jobs per chunk.
   - It must not affect model pruning, best-epoch selection, checkpoint cleanup, or the success criteria of held-out validation.

2. **Standalone validation/testing judging jobs.**
   - Training rollout judging is now wired as a standalone CPU pass in the one-shot pipeline launcher.
   - Held-out validation now supports explicit `--phase rollout`, `--phase judge`, and `--phase score`; `all` remains available as a compatibility path.
   - Testing now supports explicit `--phase rollout`, `--phase judge`, and `--phase score`; `all` remains available as a compatibility path.
   - The one-shot pipeline launcher schedules held-out validation as rollout GPU -> judge CPU -> score CPU.
   - For `num_oneshot_epochs = N`, held-out validation performs `N + 1` validations: the base model as epoch `0`, then trained checkpoints `1..N`.
   - Testing has separate GPU and CPU SLURM scripts for rollout versus judging/scoring.

3. **Checkpoint pruning.**
   - Checkpoint pruning is a separate manual CPU job (`bin_oneshot_prune`), not part of the default one-shot pipeline launcher.
   - Run pruning only after held-out validation, diagnostic training-chunk validation, testing, and any manual inspection that needs intermediate epoch checkpoints are complete.
   - This avoids deleting checkpoints that diagnostic validation or follow-up testing still needs.

4. **Next implementation step.**
   - Run a small end-to-end dry submission on Delta using one chunk and a short walltime to validate path conventions and dependency ordering.
   - If the dry run passes, migrate active experiment submissions to `scripts/hpc/bin_oneshot_pipeline.py`.

## Chunked Pipeline Smoke Test — Submitted 2026-08-04

Purpose: verify the new chunked one-shot pipeline end to end before using it for serious experiments.

Setup:

- Model: Qwen2.5 no-tool.
- Method: GRPO terminal-reward baseline.
- Training: single-GPU LoRA, rank `32`, learning rate `1e-6`.
- Chunking: `2` epochs/chunks, `100` raw questions per chunk.
- Training time: `300` seconds per epoch.
- Validation: held-out validation total epochs `2`, plus diagnostic training-chunk validation over all chunks.
- Optimizer switches: `use_adam_state = true`, `use_lr_warmup = true`.
- Configs:
  - `config/oneshot_rollout/qwen25_rollout_grpo_notool_chunk_smoke.toml`
  - `config/oneshot_generation/qwen25_generate_grpo_notool_chunk_smoke.toml`
  - `config/oneshot_train/qwen25_train_grpo_notool_chunk_smoke_lora_r32_lr1e6_2ep.toml`

Pre-submit login smoke:

- Passed locally and on the Delta login node.
- The smoke caught and fixed two trivial blockers before SLURM submission:
  - Diagnostic training-chunk validation initially rejected `validation_total_epochs`.
  - Prune initially used an underspecified training-config schema.

Submitted SLURM chain:

- Rollout: `20861708`, 1 A100, 32 CPUs, 32G, `00:50:00`, account `bfsl-delta-gpu`, partition `gpuA100x4`, pending reason `Priority`.
- Training judging: `20861709`, CPU, 32 CPUs, 32G, `01:00:00`, dependency `afterok:20861708`.
- Generation: `20861710`, CPU, 32 CPUs, 32G, `00:17:00`, dependency `afterok:20861709`.
- Training: `20861711`, 1 A100, 32 CPUs, 32G, `00:50:00`, dependency `afterok:20861710`.
- Diagnostic chunk rollout: `20861712`, 1 A100, 32 CPUs, 32G, `00:50:00`, dependency `afterok:20861711`.
- Diagnostic chunk judging: `20861713`, CPU, 32 CPUs, 32G, `00:50:00`, dependency `afterok:20861712`.
- Diagnostic chunk scoring: `20861714`, CPU, 32 CPUs, 32G, `00:50:00`, dependency `afterok:20861713`.
- Held-out validation rollout: `20861715`, 1 A100, 32 CPUs, 32G, `00:50:00`, dependency `afterok:20861711`.
- Held-out validation judging: `20861716`, CPU, 32 CPUs, 32G, `00:50:00`, dependency `afterok:20861715`.
- Held-out validation scoring: `20861717`, CPU, 32 CPUs, 32G, `00:50:00`, dependency `afterok:20861716`.
- Prune: `20861718`, CPU, 32 CPUs, 32G, `00:50:00`, dependencies `afterok:20861717` and `afterok:20861714`.

3. **Legacy inline-judgment cleanup.**
   - New downstream generation/accuracy paths use `TreeJudged`.
   - Legacy `JudgeAnswer` actions and old judged action-log replay remain for debug/fallback and should be removed or explicitly gated after the new pipeline is smoke-tested.

## Serious Chunked Training Rollout Schedule — Planned 2026-08-05

Purpose: collect larger chunked training rollouts while keeping GPU queue pressure controlled.

Batching and rollout policy:

- Extend the four Qwen2.5 serious training rollouts to `30` chunks:
  - GRPO no-tool: `250` questions/chunk, `30` chunks.
  - GRPO tool: `250` questions/chunk, `30` chunks.
  - Tree no-tool: `125` questions/chunk, `30` chunks.
  - Tree tool: `125` questions/chunk, `30` chunks.
- Add Qwen3-34B, Gemma, and Mistral no-tool training rollouts for GRPO and Tree:
  - GRPO no-tool: `250` questions/chunk, `10` chunks.
  - Tree no-tool: `125` questions/chunk, `10` chunks.
- The chunk sizes are chosen so that, without question filtering, each chunk corresponds to roughly `2000` generated leaves/trajectories for both GRPO (`8` leaves/question) and Tree (`16` leaves/question).
- Use vLLM backend, one A100, HDD result storage, and a requested SLURM walltime of `4:00:00` for each rollout job.
- Maintain at most three newly submitted rollout jobs at a time; submit follow-up jobs as earlier rollout jobs finish or fail.
- Before each serious rollout submission, run `bin_oneshot_rollout --login-smoke` on the Delta login node and submit only if it passes.

Submission status:

- Login-smoke passed on Delta for all ten planned rollout configs.
- First submitted batch, respecting the three-new-rollout cap:
  - `20883076`: Qwen2.5 GRPO no-tool, `30` chunks, `4:00:00`, 1 A100, 32 CPUs, 32G, pending for resources.
  - `20883077`: Qwen2.5 Tree no-tool, `30` chunks, `4:00:00`, 1 A100, 32 CPUs, 32G, pending for resources.
  - `20883078`: Qwen2.5 GRPO tool, `30` chunks, `4:00:00`, 1 A100, 32 CPUs, 32G, pending for resources.
- Remaining rollout queue, to submit as slots free:
  - Qwen2.5 Tree tool, `30` chunks.
  - Qwen3-34B GRPO no-tool, `10` chunks.
  - Qwen3-34B Tree no-tool, `10` chunks.
  - Gemma GRPO no-tool, `10` chunks.
  - Gemma Tree no-tool, `10` chunks.
  - Mistral GRPO no-tool, `10` chunks.
  - Mistral Tree no-tool, `10` chunks.
- After all four Qwen2.5 30-chunk rollouts complete, append the existing Qwen2.5 GRPO/Tree no-tool Adam and SGD/no-warmup training runs from `10` to `30` epochs using the same training config nicknames. The existing first ten checkpoints remain the prefix; generation should be rerun over the expanded 30 chunks before appending epochs `11..30`.

Deferred Qwen2.5 50-epoch extension:

- Extend the active Qwen2.5 no-tool Adam experiments from `30` to `50` epochs after the current GPU queue is empty.
- Scope: GRPO no-tool Adam and Tree no-tool Adam under the matched rank-32, learning-rate `1e-6`, Adam-state, warmup-enabled recipe.
- Do not submit while Qwen34/Gemma/Mistral validation or training jobs are still running or pending; this is intentionally deferred to avoid blocking current cross-model evidence collection.
- Before submission, update the relevant training configs to `num_oneshot_epochs = 50` and `validation_total_epochs = 50`, rerun generation if additional rollout chunks are needed, and run login-smoke on Delta.
- Held-out validation for the 50-epoch extension should use `--epoch-interval 3`; expected validation epochs are `0, 3, 6, ..., 48`.
- If checkpoint `30` and its training status are present, training should resume from epoch `31` rather than restarting.

Submission update:

- Submitted on 2026-08-08 because the GPU queue was near empty.
- GRPO no-tool Adam chain:
  - Rollout append to 50 chunks: `20941294`.
  - Training judging: `20941295`.
  - Generation refresh: `20941296`.
  - Training append to 50 epochs: `20941297`.
  - Held-out validation rollout with `--epoch-interval 3`: `20941298`.
  - Held-out validation judging: `20941299`.
  - Held-out validation scoring: `20941300`.
- Tree no-tool Adam chain:
  - Rollout append to 50 chunks: `20941301`.
  - Training judging: `20941302`.
  - Generation refresh: `20941303`.
  - Training append to 50 epochs: `20941304`.
  - Held-out validation rollout with `--epoch-interval 3`: `20941305`.
  - Held-out validation judging: `20941306`.
  - Held-out validation scoring: `20941307`.

Adaptive batching follow-up:

- Add a future preflight pass that estimates `max_batch_tokens` for single-GPU LoRA adaptive batching.
- This preflight should binary-search a safe padded batch tensor budget using synthetic samples, then apply a conservative safety multiplier.
- Until this exists, `max_batch_tokens` remains a manually configured training hyperparameter and should be treated as experimental.

Held-out validation cadence:

- Future held-out validation jobs should use `--epoch-interval 3`.
- This validates epoch `0` plus trained epochs divisible by `3`, reducing validation GPU cost while preserving coarse accuracy trends over long runs.
- The one-shot pipeline launcher defaults to this policy; override it only when a dense per-epoch curve is explicitly needed for a figure or debugging.

### SGD / No-Warmup Queue Policy — 2026-08-07

The Qwen2.5 no-tool SGD/no-warmup ablation is concluded as ineffective under the current rank-32, learning-rate `1e-6` LoRA recipe. Partial validation through approximately epoch 20/21 did not show a reliable increasing trend, and Tree SGD was below its base accuracy at the latest scored epoch. Do not schedule more SGD/no-warmup jobs unless a new optimizer-specific hypothesis is introduced.

### Offline Reference-Logprob KL Plan — 2026-08-09

Add KL regularization without loading a second reference model during training.

- Insert a new reference-logprob annotation pass after trajectory generation and before training.
- The annotation pass loads the frozen base model, runs teacher-forced inference over generated training trajectories, and stores one scalar reference log probability for each actual trajectory token.
- Store `ref_logprobs` only in the training trajectory artifact, aligned with `input_ids`, `labels`, `advantages`, and `old_logprobs`; do not enlarge rollout tree artifacts.
- Prompt and ignored-label positions use `0.0` because they are masked out by the training loss.
- Training requires `ref_logprobs` only when `kl_beta > 0`; non-KL runs with `kl_beta = 0` remain compatible with existing non-annotated trajectory artifacts.
- The KL term is a sampled-token approximation, not full-distribution KL: it constrains the generated trajectory tokens but does not account for unobserved candidate tokens.
- Future KL-enabled pipelines should run `rollout -> judging -> generation -> reference-logprob annotation -> training -> validation`.
- Implemented artifact naming keeps the original trajectory files intact: `trajectories_ref_logprobs.msgpack` for unchunked runs and `chunk_{n}_ref_logprobs.msgpack` for chunked runs.
- `bin_oneshot_training` automatically consumes the annotated artifact only when `training_hyperparameters.kl_beta > 0`.
- `scripts/hpc/bin_oneshot_pipeline.py` automatically inserts the GPU reference-logprob annotation stage when `kl_beta > 0`.

Initial KL experiment schedule:

- Qwen2.5 GRPO no-tool, rank-32 LoRA, learning rate `1e-6`, Adam with warmup, 50 epochs/chunks, `kl_beta = 0.04`.
- Qwen2.5 Tree no-tool, rank-32 LoRA, learning rate `1e-6`, Adam with warmup, 50 epochs/chunks, `kl_beta = 0.04`.
- Reuse existing 50-chunk no-tool generation artifacts; submit only reference-logprob annotation, KL training, and validation.

Tool rollout extension:

- Extend Qwen2.5 GRPO tool rollout from 30 to 50 chunks using the existing `grpo_tool_rollout_10chunk` nickname so chunk completion markers prevent rerunning finished chunks.
- Extend Qwen2.5 Tree tool rollout from 30 to 50 chunks using the existing `tree_tool_rollout_10chunk` nickname so chunk completion markers prevent rerunning finished chunks.
