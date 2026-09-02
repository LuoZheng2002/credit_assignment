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

1. **Guided branching uses forced selected branch tokens by default.**
   - One-shot rollout now defaults `force_selected_branch_token = true`; TOML fields or CLI flags can still make this explicit for auditability.
   - The forced-token path now preserves the selected token's old rollout log-probability instead of inserting it with synthetic logprob `0.0`, which is important under the clipped policy-ratio training objective.
   - The recovered qwen25 no-tool validation comparison is mixed but slightly favors the patched forced-token setup at later checkpoints, so future TreeMAPPO-style rollouts should use the strict forced-token path unless the experiment is explicitly a non-forced ablation.
   - Branch-budget ablations (`branch8` and `branch32`) should be rerun with forced-token branching and chunked rollout artifacts; older non-forced/legacy time-based branch-budget artifacts should not be reused.

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

Current interpretation:

1. Qwen2.5 no-tool remains the cleanest paper path because both GRPO and TreeMAPPO show modest positive validation signals under matched rank-32, learning-rate `1e-6` settings.
2. Qwen34 tool is a mixed/negative comparison for TreeMAPPO: the Tree checkpoint is below the matched GRPO checkpoint on aggregate held-out accuracy, although it improves on a subset of datasets.
3. Positive-advantage-only TreeMAPPO should be treated as a single low-priority no-tool ablation, not a family of paper-facing experiments. Future positive-only reruns should use chunked tree artifacts and should not reuse legacy unchunked trajectories.

## 50-Epoch Extension Policy — 2026-08-23

All paper-facing experiments should be extended to 50 epochs when the required rollout chunks exist or can be resumed safely. Submission priority:

1. Extend already-rollout-complete core comparisons first, because they only require training resume plus held-out validation.
   - Qwen2.5 tool GRPO: extend `grpo_tool_10chunk_lora_r32_lr1e6_adam_50ep_training` from 40 to 50 epochs.
   - Qwen2.5 tool TreeMAPPO: rerun `tree_tool_10chunk_lora_r32_lr1e6_adam_50ep_training` from rollout with forced selected branch tokens explicitly enabled, because the previous 40-epoch artifacts predate the forced-token default and should not be extended as if they used the new mechanism.
2. Extend strict forced-token TreeMAPPO variants next, because forced selected branch token is now the default mechanism.
3. Extend ablation runs after the core comparisons, prioritizing TEMPO-style branching, TreeRPO-style credit, TreeRL-style branching, TreeRL-style credit, and TreeRL combined. The Qwen2.5 no-tool TEMPO-style branching, TreeRPO-style credit, and TreeRL-style branching runs are queued at 50 epochs/chunks; the next fillers are TreeRL-style credit and TreeRL combined at 50 epochs/chunks, with the combined run reusing the TreeRL-style branching rollout artifacts.
4. Treat branch-budget runs as secondary until their forced-token 5-epoch pipelines finish and metadata confirms the generated trajectories look normal.
4. The next scientific decision should be whether to replicate the Qwen2.5 no-tool GRPO-vs-Tree comparison for robustness or spend the next queue slots on planned ablations such as branch budget, TEMPO-style branching, and forced-token guided branching.


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
   - Scheduling update 2026-08-18: do not include diagnostic training-chunk validation in future experiment plans by default. Future runs should use held-out validation and final testing only, unless a specific debugging hypothesis explicitly requires diagnostic rollout.

2. **Standalone validation/testing judging jobs.**
   - Training rollout judging is now wired as a standalone CPU pass in the one-shot pipeline launcher.
   - Held-out validation now supports explicit `--phase rollout`, `--phase judge`, and `--phase score`; `all` remains available as a compatibility path.
   - Testing now supports explicit `--phase rollout`, `--phase judge`, and `--phase score`; `all` remains available as a compatibility path.
   - The one-shot pipeline launcher schedules held-out validation as rollout GPU -> judge CPU -> score CPU.
   - For `num_oneshot_epochs = N`, held-out validation performs `N + 1` validations: the base model as epoch `0`, then trained checkpoints `1..N`.
   - Testing has separate GPU and CPU SLURM scripts for rollout versus judging/scoring.

3. **Checkpoint pruning.**
   - Checkpoint pruning is a separate manual CPU job (`bin_oneshot_prune`), not part of the default one-shot pipeline launcher.
   - Do not prune an experiment until all scheduled held-out validation jobs for that experiment have completed successfully.
   - Current-stage pruning is conservative: only manually prune checkpoints whose epoch is not a multiple of the held-out validation interval, normally `10`.
   - Keep validation-interval checkpoints, such as epochs `10`, `20`, `30`, ..., so training can resume and so missing validation/testing can be repaired without retraining.
   - If broader checkpoint cleanup is unavoidable, first copy the checkpoint directories to another disk and record the backup source path, destination path, timestamp, and covered epochs in this document.

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
- Validation: held-out validation total epochs `2`, plus diagnostic training-chunk validation over all chunks. This was a smoke-test/debugging choice; it is not the future default.
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
- Future held-out validation should use `--epoch-interval 10`; expected 70-epoch validation checkpoints are `0, 10, 20, ..., 70`.
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

- Future held-out validation jobs should use `--epoch-interval 10`.
- Future GPU SLURM jobs should use account/QOS `bfsl-delta-gpu` unless explicitly overridden by the user.
- This validates epoch `0` plus trained epochs divisible by `10`, reducing validation GPU cost while preserving coarse long-run accuracy trends.
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

### Non-KL Default Scheduling Plan — 2026-08-12

Current scheduling decision:

- Treat non-KL training as the default experimental path. The `kl_beta = 0.04` runs underperformed matched non-KL runs, and `kl_beta = 0.001` recovered some performance but has not shown a clear advantage over non-KL.
- Do not schedule additional KL jobs unless a specific KL hypothesis is introduced. Keep KL as a secondary ablation rather than the main paper evidence.
- Keep the default recipe fixed for new serious non-KL training unless a change is explicitly being tested: Qwen2.5-compatible vLLM pipeline, LoRA rank `32`, learning rate `1e-6`, Adam state enabled, warmup enabled, no KL, by-question chunked generation, held-out validation with `--epoch-interval 10`, and no diagnostic training-chunk validation.

Queue policy:

- Maintain up to `8` active experiment pipelines when the queue is healthy, including dependent GPU jobs, to improve throughput before the September paper deadline. Keep this cap flexible downward if jobs start failing due to quota, vLLM startup instability, or artifact-lock contention.
- Before every serious submission, run the matching launcher or binary with `--login-smoke` on the Delta login node and submit only if it passes.
- Prefer jobs that create a direct decision point over broad exploratory sweeps. Report job id, dependency, partition, time limit, GPU count, CPU count, memory, account/QOS, and pending reason after submission.
- Use non-KL validation first when both KL and non-KL checkpoints exist. KL validation should not block non-KL scheduling.

Near-term non-KL priority order:

1. **Complete matched Qwen2.5 no-tool evidence.** Keep the GRPO-vs-Tree comparison as the primary baseline pair, with matched rank `32`, learning rate `1e-6`, Adam/warmup, and 50-epoch curves where available.
2. **Complete matched Qwen2.5 tool evidence.** Extend or validate GRPO-tool and Tree-tool under the same fixed recipe so the paper can compare no-tool and tool settings under one policy.
3. **Run strict Tree mechanism ablation.** Schedule the forced-first-token Tree run as a non-KL Tree variant, because it directly tests whether explicit token-level branch enforcement helps the guided branching mechanism.
4. **Run branching-budget ablations.** Schedule branch-budget variants such as branch `8` and branch `32` after the standard Tree result is validated, using the same training recipe.
5. **Run related-method ablations.** Schedule TEMPO-style branching and TreeRPO/TreeRL-style advantage ablations only after the core GRPO-vs-Tree curves are available, so they explain the mechanism rather than delaying the main comparison.
6. **Add model-family evidence selectively.** Prioritize Qwen-family extensions first; treat Gemma and Mistral as lower priority unless the paper needs cross-model coverage. Mistral remains risky because previous runs showed collapse.

Decision rules:

- Treat best held-out accuracy gains below roughly `0.005` absolute as likely noise unless repeated across independent runs or supported by a targeted diagnostic/debugging run.
- Treat gains near or above `0.010` absolute as worth replicating, especially if nearby epochs show the same trend instead of a single isolated spike.
- If Tree and GRPO remain tied under matched settings, state the paper claim as competitive guided branching and diagnostic credit-assignment evidence rather than universal superiority.
- If tool runs improve more than no-tool runs, emphasize that the pipeline can train tool-use trajectories, but only claim a Tree advantage if Tree is clearly better than matched GRPO under the same validation cadence.

Recommended next submissions when queue capacity is available:

- Submit or continue Qwen2.5 tool GRPO and Tree non-KL validation/training to complete comparable epoch-interval-10 curves.
- Submit Qwen2.5 no-tool strict Tree with forced selected first token enabled.
- Submit Qwen2.5 no-tool branch-budget ablations after strict Tree is queued or completed.
- Submit TEMPO-style and TreeRPO/TreeRL-style ablations only if the queue has spare capacity after the above jobs.

Extension update:

- Extend the matched Qwen2.5 no-tool non-KL Adam runs from `50` to `70` epochs/chunks for both GRPO and Tree.
- Extend the matched Qwen2.5 tool non-KL Adam runs to `40` epochs for both GRPO and Tree. The tool rollout configs already target `50` chunks, so no tool rollout extension is needed for the 40-epoch training target.
- Keep held-out validation on `--epoch-interval 10`; expected no-tool validation epochs are `0, 10, 20, ..., 70`, and expected tool validation epochs are `0, 10, 20, 30, 40`.
- Extend active Qwen2.5 experiments to `70` epochs/chunks when their rollout artifacts can support the extension. For interval-10 validation, report epochs `0`, `10`, `20`, ..., `70`.

### Ablation Extension Schedule — 2026-08-15

Extend the Qwen2.5 no-tool ablations from `5` to `30` chunks/epochs under the same fixed non-KL recipe: LoRA rank `32`, learning rate `1e-6`, Adam state enabled, warmup enabled, one A100, held-out validation with `--epoch-interval 10`, and no diagnostic training-chunk validation.

- Updated the existing ablation rollout configs to `num_chunks = 30` while keeping their existing artifact nicknames so completed chunks can be reused and the jobs append chunks `5..29`.
- Updated the existing ablation training configs to `num_oneshot_epochs = 30`, `validation_total_epochs = 30`, and `total_time_limit_hours = 4.0`.
- Submitted the two highest-priority ablation extensions first to keep the runnable GPU queue at five jobs:
  - TEMPO-style branching: rollout `21178950`, judging `21178951`, generation `21178952`, training `21178953`, held-out validation rollout/judge/score `21178954`/`21178955`/`21178956`.
  - TreeRL-style advantage: rollout `21178957`, judging `21178958`, generation `21178959`, training `21178960`, held-out validation rollout/judge/score `21178961`/`21178962`/`21178963`.
- Remaining ablation extensions to submit when queue capacity opens:
  - TreeRPO-style advantage.
  - TreeRL-style branching.

Testing follow-up:

- The first serious tests for matched Qwen2.5 no-tool Adam used best validation epochs beyond epoch `20`: GRPO epoch `36` and Tree epoch `27`.
- To test whether early checkpoints generalize better, submit additional 5-rollout serious tests using the best validation epochs within the first ten epochs:
  - GRPO no-tool epoch `3`: test rollout/judge/score `21224269`/`21224270`/`21224271`.
  - Tree no-tool epoch `2`: test rollout/judge/score `21224272`/`21224273`/`21224274`.
- These use `config/rollout_config_testing_5rollouts.json`, so each testing sample is rolled out five times and reported test means are averages over those five trials per dataset.

vLLM batching follow-up:

- The CUDA 13 / vLLM `0.27.1` retry showed unusually long cold-start time and the fixed `VLLM_MAX_NUM_SEQS=32` may also cap rollout throughput after startup.
- Run one controlled serious test-rollout retry with `VLLM_MAX_NUM_SEQS=unset`, while keeping `VLLM_MAX_MODEL_LEN=4096`, `VLLM_MAX_NUM_BATCHED_TOKENS=4096`, and `VLLM_ENFORCE_EAGER=1`.
- If the uncapped vLLM scheduler run completes successfully and improves throughput without OOM, update the global SLURM defaults to omit `--max-num-seqs`; otherwise keep the explicit conservative cap.
- Follow-up controlled tests with `VLLM_MAX_NUM_SEQS=128` completed successfully for matched Qwen2.5 no-tool GRPO and Tree test rollouts. Use `128` as the default future SLURM value rather than omitting `--max-num-seqs`, because the uncapped run was less stable while `128` kept startup bounded and improved usable test-rollout coverage.

vLLM startup latency investigation:

- Some CUDA 13 / vLLM `0.27.1` jobs spent more than `1800` seconds before the wrapper health endpoint became ready.
- Initial controlled Qwen3-4B startup smokes compare minimal vLLM flags with and without `--max-num-seqs 128`, while omitting forced eager mode and per-job cold cache directories.
- Add startup timing instrumentation to future vLLM launches so logs include timestamps before entering the vLLM module, before vLLM emits its own first log line, and before the health check succeeds.
- Pending diagnosis: determine whether the dominant delay is Python import/package metadata access, vLLM initialization, model file access, CUDA/NCCL initialization, or cache compilation.
- Do not globally change cache or eager-mode policy based only on speculation; use the startup-smoke measurements to decide whether persistent cache directories, default cache behavior, or longer health timeouts are justified.
- Resolution update 2026-08-27: the failed Gemma and Qwen2.5 validation jobs timed out while importing Python/vLLM from the HDD-hosted venv. The shared vLLM runtime was rebuilt at `/u/zluo8/credit_assignment/venvs/vllm-latest-cu130`, and all future SLURM inference jobs now default `VLLM_VENV` to this `/u` path. Login-node import timing from `/u` is normal (`torch` about 15s, `triton` about 1s, `vllm` about 19s), so future startup debugging should first confirm that jobs are using the `/u` venv before changing vLLM flags.
- Follow-up update 2026-08-27: the Mistral retry submitted before the `/u` patch still used the HDD vLLM venv and failed with the same wrapper-health timeout. Experiment SLURM scripts now hardcode `VLLM_VENV=/u/zluo8/credit_assignment/venvs/vllm-latest-cu130` instead of preserving an inherited environment variable, so stale submitter environments cannot silently revert jobs to the HDD runtime.
- Pending-job audit 2026-08-27: the Qwen2.5 GRPO no-tool and Tree no-tool recovery validation rollouts were submitted before the `/u` vLLM fix and were replaced with new validation chains dependent on the running recovery training jobs. The remaining pending Gemma, Mistral, and forced-token Qwen2.5 jobs were submitted after the `/u` default was installed; the Mistral replacement and all future experiment SLURM scripts use the hardcoded `/u` vLLM runtime. The standalone vLLM smoke scripts were also hardened to use the same `/u` runtime.
- Timeout handling update 2026-08-27: the Qwen2.5 GRPO no-tool and Tree no-tool recovery trainings reached about epoch `23` but timed out under a `4h` training walltime; their configs were updated to `total_time_limit_hours = 13.0` so resumed training has enough walltime to reach epoch `70`. Gemma and Mistral 50-chunk training-rollout judging jobs timed out under the default `30m` CPU budget, so those judging stages were resubmitted with `2h` while preserving `16` CPUs and `16G` memory.
- Judge failure update 2026-08-27: Gemma 50-chunk judging failed because `deepseek/deepseek-v4-pro` returned only hidden reasoning and hit the `1024` completion-token limit before producing the required boxed verdict. The judging request now gives `deepseek/deepseek-v4-pro` `4096` completion tokens while preserving the existing fail-fast behavior for invalid judge outputs. The failed run left only a temporary JSONL output, which was removed before resubmission.

### Executed-Experiment Epoch Extension Policy — 2026-08-16

For every experiment variant that has already produced a usable executed training run, schedule or extend the matched training run to at least `20` total epochs/chunks when prerequisites and queue capacity permit.

- Apply this as a minimum floor for executed experiments, not as a replacement for stronger existing targets. Runs already scheduled beyond `20` epochs, such as the Qwen2.5 no-tool `70`-epoch runs and Qwen2.5 tool `40`-epoch runs, keep their larger targets.
- Use the fixed non-KL default recipe unless the experiment is explicitly testing another mechanism: LoRA rank `32`, learning rate `1e-6`, Adam state enabled, warmup enabled, no KL, by-question chunked generation, and vLLM inference.
- Keep held-out validation at `--epoch-interval 10` for long runs unless dense validation is specifically needed for a figure or debugging.
- Reuse existing rollout, judgment, generation, and checkpoint artifacts whenever chunk completion markers and training resume metadata show that appending is safe.
- Do not prioritize failed or scientifically deprecated settings such as SGD/no-warmup unless a new hypothesis makes them relevant again.

Submission update — 2026-08-18:

- Updated future SLURM vLLM defaults to `VLLM_MAX_NUM_SEQS=128` after controlled Qwen2.5 no-tool test rollouts completed successfully.
- Submitted four additional planned non-KL pipelines after login-smoke checks passed:
  - Qwen2.5 GRPO tool 40-epoch pipeline: rollout/judge/generation/training/chunk-validation/held-out validation jobs `21268041`-`21268050`.
  - Qwen2.5 Tree tool 40-epoch pipeline: rollout/judge/generation/training/chunk-validation/held-out validation jobs `21268051`-`21268060`.
  - Qwen2.5 TreeRPO-style advantage 30-epoch ablation: rollout/judge/generation/training/chunk-validation/held-out validation jobs `21268061`-`21268070`.
  - Qwen2.5 TreeRL-style branching 30-epoch ablation: rollout/judge/generation/training/chunk-validation/held-out validation jobs `21268071`-`21268080`.

### Empty Training-Chunk Policy — 2026-08-19

Chunk-backed one-shot training must fail fast if any epoch's trajectory chunk contains zero training trajectories.

- An empty training chunk means every question in that chunk was filtered out, typically because the rollout/judgment outcomes were all correct or all incorrect with no mixed correctness signal.
- Do not silently carry the previous model forward or count the epoch as trained; this is misleading for validation curves and downstream model selection.
- Treat an empty chunk as an investigation blocker: inspect per-chunk correctness, judgment cache behavior, and rollout answer diversity before resubmitting training.
- If empty trajectory chunks are plausibly caused by corrupted or legacy judgment-cache behavior, do not rerun the expensive initial training rollout. Preserve completed rollout tree chunks, back up and clear stale judgment and trajectory-generation artifacts for the affected experiment, rerun judging for the entire experiment, rerun trajectory generation from scratch, then continue with training and validation.
- Generation may still emit explicit empty chunk files so the failure identifies the exact epoch/chunk rather than appearing as a missing-artifact bug.

### Judgment API Failure Policy — 2026-08-19

OpenRouter/API failures during judging must be fatal and must not be converted into incorrect verdicts.

- Previous chunked judging behavior returned a synthetic `false` verdict after repeated judge-model failures, which could poison the cache as `phase1_unanimous` incorrect records.
- This is especially dangerous because phase-1 unanimous records intentionally omit raw judge outputs, so later cache hits cannot distinguish real unanimous incorrect judgments from fallback failures.
- If insufficient credit, quota exhaustion, rate limiting, payment-required errors, or repeated invalid judge responses occur, the judging job should exit nonzero after flushing already completed cache writes.
- Cache recovery should be selective: rewrite or delete affected JSONL records in the relevant cache chunks rather than deleting the entire cache directory.

Update — 2026-08-19:

- Exact string matches between the model answer and reference answer are now judged locally as `exact_match` correct before any API judge is called.
- Judge-model outputs that do not provide an explicit boxed `correct` or `incorrect` verdict are treated as invalid attempts. After three invalid/error attempts for a model, the judging job exits immediately with a logged reason and no cached verdict for that request.
- Because earlier fallback behavior may have cached unreliable incorrect and correct verdicts, the current experiment judgment caches should be backed up and cleared before rerunning affected judging jobs.

### Testing Rejudge Focus — 2026-08-19

Current priority is to rejudge preserved testing rollouts using the corrected judging semantics and per-phase judge-model parallelism.

- Canceled the active Qwen2.5 forced-token rollout/training pipeline jobs `21292741`-`21292747` to avoid competing for attention while testing judgments are being repaired.
- Only three configured testing rollouts currently have reusable tree artifacts without rerunning rollout:
  - Qwen2.5 GRPO no-tool early best, epoch `3`, five explicit rollout trials: judge/score jobs `21294020`/`21294021`.
  - Qwen2.5 GRPO no-tool long-run best, epoch `36`, legacy single artifact with five trunks: judge/score jobs `21294022`/`21294023`.
  - Qwen2.5 Tree no-tool long-run best, epoch `27`, legacy single artifact with five trunks: judge/score jobs `21294024`/`21294025`.
- Before resubmission, backed up and cleared the selected testing judgment outputs, test-accuracy JSON files, and the affected experiment judgment-cache directories under `/work/hdd/bhph/zluo8/credit_assignment/results/cache_backups/testing_rejudge_parallel_backup_20260819_120043`.
- Rejudge configs are stored under `config/testing_rejudge/`; `rollout_config_testing_legacy5.json` is only for scoring older single-artifact test rollouts that contain five trunks per tree.

### Independent Tree Judge/Score Program — 2026-08-19

Judging and scoring should no longer be implemented separately inside rollout binaries.

- Use `bin_tree_judge_score` for CPU-only tree judging and scoring across training-chunk diagnostics, held-out validation, and testing.
- The shared implementation lives in `src/tree_judge_score.rs` and handles the common tree-artifact to judgment-output path.
- Testing remains explicitly distinct because modern testing can use multiple rollout trials. Use `--num-rollout-trials N` for per-trial directories and omit it for legacy single-artifact testing rollouts.
- For legacy single-artifact testing rollouts that contain multiple trunks per tree, pass the matching `--num-trunks` value when scoring.
- Use `--phase judge-score` for a single CPU task that judges and immediately scores; use `--phase judge` or `--phase score` only for recovery/debugging.
- Delta SLURM wrapper: `slurm/tree_judge_score.slurm` requests `16` CPUs, `16G` memory, and CPU partition by default.
- `bin_run_test` is now rollout-only at the CLI level (`all` and `rollout` only). Validation and diagnostic rollout launchers should be migrated next to submit `bin_tree_judge_score` jobs for judge/score rather than using their embedded legacy phases.

### Training Rollout Rejudge and Regeneration — 2026-08-19

To recover from unreliable legacy judgment-cache behavior, preserve completed training rollout tree artifacts but discard downstream judgments and generated trajectories before reuse.

- Legacy note: a broad cleanup previously removed trained checkpoint/model artifacts under `large_files/*/*/oneshot_epoch_*` to free disk space. Do not repeat this policy for active experiments; it can make later interval validation or training resume impossible.
- For current runs, preserve completed model checkpoints until all planned validation and testing for the experiment are done. If space pressure requires action, manually prune only non-validation-interval epochs, or back up checkpoints to another disk before deletion and record the backup here.
- Rejudge completed training rollout tree artifacts with the corrected chunked judging semantics, then regenerate training trajectories from those fresh `TreeJudgment` JSONL files.
- Run these as CPU-only two-stage pipelines: `bin_oneshot_judging` over `trees_training_oneshot`, followed by `bin_oneshot_generation` for the matching generation config.
- Maintain up to six active/pending rejudge-regeneration pipelines at a time. Generation jobs should depend on their corresponding judging jobs with `afterok`.
- If a generation config points at stale or missing rollout artifacts, skip it and record the missing prerequisite rather than fabricating a replacement mapping.
- Update: replace the phase-1 judge model `qwen/qwen3-32b`, which repeatedly returned answer-like text instead of boxed correctness verdicts, with `openai/gpt-5-nano`. GPT-5 Nano judging requests use OpenRouter `reasoning.effort = "minimal"`.

Update — 2026-08-19 later:

- Judging failure diagnostics now include the full question, model answer, reference answer, base prompt, and per-attempt raw judge outputs when a judge model fails after all retries. This is specifically to diagnose `openai/gpt-5-nano` cases that return a boxed answer such as `\text{None of these}` instead of `correct` or `incorrect`.
- Submitted five Qwen2.5 20-epoch non-KL pipelines after login-smoke checks passed. All use LoRA rank `32`, learning rate `1e-6`, Adam state enabled, warmup enabled, vLLM inference, and held-out validation with `--epoch-interval 10` for epochs `0`, `10`, and `20`:
  - GRPO no-tool: jobs `21300818`-`21300824`.
  - TreeMAPPO no-tool: jobs `21300825`-`21300831`.
  - GRPO tool: jobs `21300832`-`21300838`.
  - TreeMAPPO tool: jobs `21300839`-`21300845`.
  - TreeMAPPO no-tool with forced first branch token: jobs `21300846`-`21300852`.

### Stage Log and Metadata Separation — 2026-08-19

Pipeline stages should keep procedural logs and statistical metadata separate.

- Logs are for verbose procedural events and failure diagnostics. They should live under the experiment/stage path in `small_files/<model>/<config>/...` or as SLURM stdout/stderr for the corresponding job, not in shared `/tmp` locations.
- Metadata is JSON and is for concise stage statistics and representative examples. It must be keyed by experiment configuration and stage so a job from another experiment or phase cannot overwrite it.
- Judging metadata is written next to the judging output JSONL by default as `*.metadata.json`. It includes total judged answers, exact matches, cache hits/misses, cache-hit and cache-miss correctness counts/ratios, overall correctness counts/ratio, model throughput summary, and 10 deterministic pseudo-random question-answer samples with raw judge outputs, parsed outputs, and final verdicts.
- Scoring metadata is written next to the score JSON by default as `*.metadata.json` and includes the score object plus the tree artifact and judgment paths used to produce it.
- One-shot training-set generation metadata is written as `oneshot_generation_metadata.json` next to `training_trajectories_stats.json`. It includes stage/config paths plus up to 10 decoded tree examples and up to 10 decoded trajectory examples for quick agent inspection.
- Rollout metadata is written as `oneshot_rollout_metadata.json` next to the rollout summary. It includes rollout config paths, chunking settings, forced-token setting, summary/timing paths, and up to 10 decoded tree examples.
- Training metadata is written as `oneshot_training_metadata.json` under the one-shot training summary directory. It preserves per-epoch sample counts, cumulative iterations, throughput, longest accepted trajectory length, optimizer choices, training limits, checkpoint root paths, and resume start epoch. Validation/testing metadata should use the same judge/score metadata schema as the independent tree judge-score program.
- After each experiment stage finishes, inspect the corresponding metadata JSON before treating the result as usable. Check that counts, cache hit rates, correctness ratios, throughput, trajectory/tree examples, and sampled judge outputs have no visible anomaly.

### Held-Out Validation Trials — 2026-08-19

Held-out validation should support stacked rollout trials, matching the testing pipeline's retry/extension pattern.

- Set `validation_num_rollout_trials = 3` in one-shot training configs by default and keep validation epoch interval at `10` unless a run has a specific reason to deviate.
- Multi-trial validation stores artifacts under per-epoch trial directories, for example `epoch_10/trees_validation_oneshot/trial_0`, `trial_1`, and `trial_2`.
- Increasing `validation_num_rollout_trials` later appends new trial directories without redoing completed lower-index trials whose chunk done markers are present.
- Validation scoring aggregates weighted counts across all available completed trials for each requested epoch before writing the ordinary one-shot training validation summary.
- Multi-trial held-out validation must be completion-based, not time-limited: each requested trial should finish all held-out questions before the rollout phase exits. Judge/score phases should only accept a trial when every expected validation chunk has a done marker; a single partial marker is not sufficient.

### Scheduling Override — Tree Scope and Forced-Token Gate — 2026-08-19

Near-term scheduling should reduce tree-method breadth until the forced-first-token question is resolved.

- Put off new TreeMAPPO/tree-method experiments outside the Qwen2.5 no-tool setting. For Qwen34, Gemma, Mistral, Llama, and other non-Qwen2.5 or tool settings, schedule GRPO-only runs unless a specific result creates a new reason to revisit tree methods.
- Keep Qwen2.5 no-tool as the primary tree comparison setting because it is the cleanest matched GRPO-vs-Tree evidence currently available.
- Highest tree priority: finish the Qwen2.5 no-tool forced-selected-first-token pipeline and compare it against the matched non-forced Qwen2.5 no-tool Tree run under the same LoRA rank `32`, learning rate `1e-6`, Adam/warmup recipe, validation interval `10`, and three held-out validation trials.
- Decision gate: if forced first token improves Tree validation/testing accuracy or stability without clear metadata anomalies, make it the default for future TreeMAPPO experiments. If it is neutral or worse, keep it disabled and treat it as an ablation result.
- Do not queue branch-budget, TEMPO, TreeRPO, TreeRL, or non-Qwen tree extensions ahead of the forced-token decision unless the queue would otherwise sit idle and GRPO-only work is already covered.

Update — 2026-08-20:

- The previous forced-token ablation is treated as invalid because the forced token was inserted after generation rather than included in the prompt before generating the continuation.
- A patched fresh Qwen2.5 no-tool forced-token pipeline was submitted from training rollout with new artifact nicknames:
  - Rollout/generation/training configs: `qwen25_rollout_tree_notool_forced_patched_30chunk.toml`, `qwen25_generate_tree_notool_forced_patched_30chunk.toml`, and `qwen25_train_tree_notool_forced_patched_lora_r32_lr1e6_30ep_adam.toml`.
  - Jobs: rollout `21329601`, training-rollout judging `21329602`, generation `21329603`, training `21329604`, held-out validation rollout/judge/score `21329605`/`21329606`/`21329607`.
- After the patched forced-token generation finishes, inspect `oneshot_rollout_metadata.json` and `oneshot_generation_metadata.json` tree/trajectory examples before treating the result as usable. Specifically check that forced-branch starts no longer show obvious inserted-token artifacts such as duplicated words (`can can`, `into into`, `expression expression`), malformed prefixes (`SubSub...`), or stray LaTeX fragments introduced at branch boundaries.
- Until this patched forced-token decision is resolved, postpone other tree-related experiments. Available queue capacity should go to GRPO-only recovery or extension jobs, especially non-Qwen2.5 model validation/retry work.

### GRPO 40-Epoch Extension — 2026-08-20

Extend the GRPO-only pipelines to `40` epochs/chunks under the fixed recipe: LoRA rank `32`, learning rate `1e-6`, Adam state enabled, warmup enabled, vLLM inference, held-out validation with `--epoch-interval 10`, and three validation rollout trials.

- Qwen2.5 GRPO no-tool: append existing `grpo_notool_10chunk_lora_r32_lr1e6_adam_20ep_training` from epoch `20` to `40`; rollout config already has at least `40` chunks.
- Qwen2.5 GRPO tool: append existing `grpo_tool_10chunk_lora_r32_lr1e6_adam_20ep_training` from epoch `20` to `40`; rollout config already has at least `40` chunks.
- Qwen3-4B GRPO no-tool: extend `grpo_notool_rollout_10chunk` and `grpo_notool_generation_10chunk` to `40` chunks, then append `grpo_notool_10chunk_lora_r32_lr1e6_10ep_training` to epoch `40`.
- Qwen3-4B GRPO tool: start a fresh chunked `40`-chunk pipeline with `grpo_tool_rollout_40chunk`, `grpo_tool_generation_40chunk`, and `grpo_tool_40chunk_lora_r32_lr1e6_40ep_training`.

### Orthogonal Tree Ablation Matrix — 2026-08-22

Future tree-method ablations should be named and scheduled by two independent axes: the branching algorithm used during rollout and the advantage algorithm used during trajectory generation. Do not encode a paper name as a monolithic condition unless both axes are intentionally set to that paper-style combination.

Branching algorithm abbreviations:

| Abbrev. | Code policy / switch | Meaning | Implementation status |
| --- | --- | --- | --- |
| `FLAT` | `TreeMappoGuided` with `num_trunks == num_leaves` | Flat independent rollouts; no tree branching is exercised. | Existing GRPO rollout pattern. |
| `TMB` | `TreeMappoGuided` | TreeMAPPO guided branch-point selection using structural penalties and alternate-token probability. | Existing. |
| `TMBF` | `TreeMappoGuided` plus `--force-selected-branch-token` | TreeMAPPO guided branching that forces the selected alternate first token in the new branch. | Existing patched path; decision pending from Qwen2.5 no-tool result. |
| `TPB` | `TempoSpontaneous` | TEMPO-style prefix-tree branching from spontaneous divergence among sampled rollouts. | Existing approximation. |
| `TRLEB` | `TreeRlEntropyGuided` | TreeRL-style entropy-guided branch-point selection using top-k entropy as branch score. | Existing approximation. |
| `TRPOB` | not implemented | TreeRPO fixed-step tree sampler with branches after fixed segment lengths. | New feature if needed for faithful TreeRPO sampler ablation. |
| `TPOB` | not implemented | TreePO-style segment-budget tree sampler with dynamic branch allocation / fallback. | Future-only unless TreePO becomes a primary comparison. |

Advantage algorithm abbreviations:

| Abbrev. | Code policy | Meaning | Implementation status |
| --- | --- | --- | --- |
| `GRPOA` | `GrpoTerminalReward` | GRPO-style group-normalized terminal reward assigned to flat response trajectories. | Existing; requires `num_trunks == num_leaves`. |
| `TMA` | `TreeMappoPosterior` | TreeMAPPO posterior segment advantage from probabilistic MAP credit assignment. | Existing. |
| `TRPOA` | `TreeRpoWinRate` | TreeRPO-style child-group outcome comparison after backing terminal correctness into subtree values. | Existing approximation; use as TreeRPO advantage-only ablation. |
| `TRLA` | `TreeRlLocalGlobal` | TreeRL-style local-global subtree value difference advantage. | Existing approximation. |
| `TPA` | not implemented | TEMPO-style prefix TD / branch-gated advantage correction. | New feature if a faithful TEMPO credit ablation is needed. |
| `TPOA` | not implemented | TreePO-style ancestor/subgroup relative segment advantage with global normalization. | Future-only unless TreePO becomes a primary comparison. |

Recommended matrix for paper-relevant experiments:

| Branching | Advantage | Experiment meaning |
| --- | --- | --- |
| `FLAT` | `GRPOA` | GRPO baseline. |
| `TMB` | `TMA` | Main TreeMAPPO method. |
| `TMBF` | `TMA` | Forced-token TreeMAPPO mechanism ablation. |
| `TPB` | `TMA` | Branching-only TEMPO-style ablation against TreeMAPPO credit. |
| `TMB` | `TRPOA` | TreeRPO advantage-only ablation under the same guided tree. |
| `TMB` | `TRLA` | TreeRL advantage-only ablation under the same guided tree. |
| `TRLEB` | `TMA` | TreeRL branching-only ablation under TreeMAPPO credit. |
| `TRLEB` | `TRLA` | TreeRL-style combined branching/advantage ablation. |

Implementation note:

- `bin_oneshot_rollout` now accepts `--branching-policy` as an explicit rollout-time override, separate from the rollout JSON file.
- `bin_oneshot_generation` now accepts `--training-advantage-policy` as an explicit generation-time override, separate from the TOML file.
- `scripts/hpc/bin_oneshot_pipeline.py` passes these as `--branching-policy` and `--training-advantage-policy`, and the rollout/generation SLURM scripts forward extra arguments to the Rust binaries.
- Rollout and generation metadata now record the resolved algorithm names and abbreviations so completed artifacts can be audited without reconstructing CLI flags.
- Branching and advantage are decoupled for all currently implemented algorithms except where the algorithm itself imposes structural constraints, such as `GRPOA` requiring flat rollout (`num_trunks == num_leaves`).
- The only new algorithmic features required for the full literature-faithful matrix are `TRPOB`, `TPA`, and optionally `TPOB`/`TPOA`. These should not block the immediate orthogonal ablation matrix above because the currently needed paper controls can be expressed by rearranging existing `TMB`, `TMBF`, `TPB`, `TRLEB`, `TMA`, `TRPOA`, `TRLA`, and `GRPOA` logic.

### Ablation Submission Priority — 2026-08-22

Submit orthogonal ablation combinations in this order, skipping combinations that are already completed or active in SLURM/artifacts:

1. `TMBF` + `TMA`: forced-token TreeMAPPO mechanism ablation. Highest priority because it tests whether the branch token selected by the guided score should be enforced during generation.
2. `TPB` + `TMA`: TEMPO-style branching-only control. This isolates spontaneous prefix-tree exploration while keeping TreeMAPPO credit assignment.
3. `TMB` + `TRPOA`: TreeRPO advantage-only control. This isolates child-group outcome comparison under the same guided tree.
4. `TMB` + `TRLA`: TreeRL advantage-only control. This isolates local-global subtree value credit under the same guided tree.
5. `TRLEB` + `TMA`: TreeRL entropy-branching-only control. This isolates entropy-guided branching under TreeMAPPO credit.
6. `TRLEB` + `TRLA`: combined TreeRL-style control. This tests the currently implemented TreeRL-style branching and advantage pair.
7. `FLAT` + `GRPOA`: GRPO baseline. Keep as a matched baseline rather than a tree-method ablation; skip when matched GRPO already exists.
8. `TMB` + `TMA`: main TreeMAPPO method. Skip as an ablation when the main method already exists.
9. Branch-budget variants are secondary mechanism checks and should run after the above matrix unless a specific paper claim depends on them.
10. Positive-only TreeMAPPO is retained only as one low-priority Qwen2.5 no-tool ablation because it appears in the held-out test table. Do not schedule tool positive-only or repeated positive-only variants. If rerun, start from chunked tree artifacts and regenerate chunked trajectories; do not use legacy unchunked positive-only training sets.

Current implementation status:

- `TMBF` + `TMA` is already active for Qwen2.5 no-tool and Qwen2.5 tool patched forced-token pipelines.
- `TMB` + `TRPOA` and `TRLEB` + `TMA` have completed 30-epoch Qwen2.5 no-tool jobs in recent SLURM history; do not resubmit unless metadata later shows a correctness anomaly.
- `TPB` + `TMA`, `TMB` + `TRLA`, and `TRLEB` + `TRLA` are the next missing Qwen2.5 no-tool ablations to schedule.
- `TRLEB` + `TRLA` uses the existing TreeRL entropy rollout and a separate generation/training nickname so it does not overwrite the branching-only trajectory artifacts.

### Validation Cleanup Priority — 2026-08-26

Future scheduling should prioritize missing held-out validation records before launching new training or rollout experiments. Stale validation entries have been pruned so retained summaries should only include epoch `0` or multiples of `10`; epoch `30` and `60` are retained only when the log confirms `6` validation trials.

Checkpoint retention rule:

- Do not prune active experiment checkpoints until all planned validation jobs for that experiment have completed.
- Manual pruning at this stage should only delete epochs that are not multiples of the active validation interval, currently `10`.
- Preserve epoch multiples of `10` even if they are not the current best model, because they are needed for interval validation, testing repair, and append/resume workflows.
- Before any broader cleanup, back up checkpoints to a different disk when possible and record the backup path, source path, epoch range, and date in this document.
- Recovery update 2026-08-27: Qwen2.5 GRPO no-tool and Tree no-tool non-forced checkpoint directories had been removed, so their missing validation epochs cannot be repaired by validation alone. Their 70 generated trajectory chunks are intact, so the repair path is to retrain from the existing trajectory chunks and then run 6-trial interval-10 held-out validation. The current validation code reports `deepmath`, `math`, and `numinamath`, so rerun summaries should not repeat the legacy missing-NuminaMath issue.
- Non-Qwen2.5 scheduling rule 2026-08-27: keep non-Qwen2.5 GRPO extensions at `50` epochs/chunks rather than the Qwen2.5 `70`-epoch target. Gemma and Mistral GRPO no-tool are extended from `30` to `50` epochs/chunks as secondary queue-fill work while Qwen2.5 no-tool validation recovery is prioritized.

Immediate priority:

1. Re-run 6-trial interval-10 held-out validation for Qwen2.5 GRPO no-tool, filling epochs `30`, `40`, `50`, `60`, and `70`.
2. Re-run 6-trial interval-10 held-out validation for Qwen2.5 Tree no-tool non-forced, filling epochs `30`, `40`, `50`, `60`, and `70`.
3. Finish or re-run Qwen2.5 Tree tool forced-token validation for epochs `60` and `70` with 6 trials.
4. Upgrade Qwen2.5 GRPO tool and Tree tool non-forced validation to 6 trials if they are used in a direct forced-token comparison; otherwise label them explicitly as 3-trial results.
5. Let already submitted Branch-8, Branch-32, TreeRL-branching, and TreeRL-advantage pipelines finish before scheduling additional ablation training.

Do not use pruned interval-3 best epochs such as `27`, `51`, `63`, `66`, or `69` in paper tables unless they are explicitly labeled as legacy exploratory validation.

### Queue Repair and Serious-Test Fill — 2026-09-01

Failed-job handling:

- Canceled stale dependency jobs from the Llama Tree no-tool validation timeout and the Mistral Tree no-tool validation-judging timeout.
- Resubmitted Mistral Tree no-tool validation judging with a `2h` CPU walltime and a dependent score job: `21712559` -> `21712560`.
- Resubmitted Llama Tree no-tool validation rollout with an `8h` GPU walltime and dependent judge/score jobs: `21712561` -> `21712562` -> `21712563`.

Queue fill:

- Filled the Delta GPU queue to `8` active roots with one repaired validation rollout plus seven best-validation-epoch serious-test rollout pipelines.
- Serious tests use `5` rollout trials, `vllm`, one A100, `32` CPUs, `32G` memory, and `2h` rollout walltime unless later logs show that a model needs a larger testing budget.
- Submitted serious-test chains:
  - Qwen2.5 tool GRPO, best validation epoch `30`: `21712589` -> `21712590` -> `21712591`.
  - Qwen2.5 tool TreeMAPPO, best validation epoch `70`: `21712592` -> `21712593` -> `21712594`.
  - Qwen2.5 no-tool forced-token TreeMAPPO, best validation epoch `50`: `21712595` -> `21712596` -> `21712597`.
  - Qwen2.5 tool forced-token TreeMAPPO, best validation epoch `50`: `21712598` -> `21712599` -> `21712600`.
  - Qwen3-4B no-tool GRPO, best validation epoch `40`: `21712601` -> `21712602` -> `21712603`.
  - Qwen3-4B tool GRPO, best validation epoch `40`: `21712604` -> `21712605` -> `21712606`.
  - Gemma no-tool GRPO, best validation epoch `50`: `21712607` -> `21712608` -> `21712609`.

Future scheduling implication:

- Best-epoch serious testing is now an explicit part of the queue-fill policy after validation exists. Prefer ready serious tests over low-priority exploratory reruns when the GPU queue is underfilled.
- The low-priority positive-only ablation remains deferred; if it is rerun, it must use chunked tree artifacts and should not reuse legacy unchunked trajectories.

### Additional Serious-Test Fill — 2026-09-01

After the first serious-test rollouts began completing, the active GPU-root count dropped to `5`. To restore the queue to `8`, submitted three additional ready Qwen2.5 no-tool ablation serious-test pipelines after checkpoint checks and `bin_run_test --login-smoke` passed:

- TEMPO-style branching ablation, best validation epoch `70`: `21716392` -> `21716393` -> `21716394`.
- TreeRPO-style advantage ablation, best validation epoch `40`: `21716395` -> `21716396` -> `21716397`.
- TreeRL advantage-only ablation, best validation epoch `60`: `21716398` -> `21716399` -> `21716400`.

All three testing rollout roots request `bfsl-delta-gpu`, `gpuA100x4`, one A100, `32` CPUs, `32G` memory, and `2h` walltime. Their dependent judge jobs request `bfsl-delta-cpu`, `16` CPUs, `16G`, and `2h`; score jobs request `30m`.

Additional queue top-up:

- The queue dropped to `7` while earlier serious-test rollouts completed, so one more ready ablation serious-test pipeline was submitted.
- TreeRL branching-only ablation, best validation epoch `20`: `21716439` -> `21716460` -> `21716461`.
- This job uses the same serious-test resource policy: one A100, `32` CPUs, `32G`, `2h` rollout walltime, with dependent CPU judge/score jobs.
