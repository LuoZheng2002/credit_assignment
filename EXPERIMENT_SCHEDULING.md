# Experiment and Scheduling Plan

Last updated: 2026-07-19

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
- Safe rerun job: `20280442`, `training_mistral_r16_safe`, pending `Priority`
- Safe rerun config: `config/oneshot_train/mistral_train_grpo_notool_lora_r16_5m_safe.toml`
- Training data: `/work/hdd/bhph/zluo8/credit_assignment/results/medium_files/mistral/grpo_notool_generation_1h/training_trajectories/trajectories.msgpack`
- Safe rerun resources: `gpuA100x4`, `bfsl-delta-gpu`, `1 x A100`, `32` CPUs, `32G` memory, walltime `01:39:00`
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

### Slot 1: Active Mistral LoRA safe rerun

- job `20280442`
- monitor for explicit `synthetic_oom_preflight_*` logs, whether cutoff `2500` avoids OOM, and how many epochs finish.

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

- queue additional single-GPU LoRA experiments only after current Mistral/Qwen34 signals clarify whether adapter persistence is healthy.
- the preferred branch remains a LoRA rank sweep around the already-successful no-tool GRPO setup.

Suggested initial rank sweep:

- rank 8
- rank 16
- rank 32

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

Rationale:

- Rank 8 tests whether we can keep most of the gain with a smaller adapter and lower memory pressure.
- Rank 16 is the known-good anchor and should remain in the sweep for direct comparison.
- Rank 32 tests whether extra adapter capacity buys measurable validation gain.

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

As of now, the most important scientific and scheduling conclusions are:

1. **The no-tool GRPO split pipeline works.**
2. **LoRA no-tool GRPO already gives a credible positive signal.**
3. **The next highest-value uncertainty is the best single-GPU LoRA recipe, including LoRA rank.**
4. **Full-parameter no-tool training is stashed as a backup path while LoRA is debugged and optimized.**
5. **Mistral rank 16 has not produced a serious result yet, but the partial epoch-1 validation did not show immediate catastrophic collapse.**
6. **Qwen34 rollout is the main active long-running pipeline and is progressing normally.**
7. **VERL is currently an isolated integration smoke, not scientific evidence; its latest blocker was batch-size divisibility in the 3-training-GPU FSDP actor pool.**
8. **We should not run orchestrator-scale jobs until the LoRA recipe is stabilized.**
9. **The next paper-central milestone is the first valid no-tool GRPO vs TreeMAPPO comparison on Qwen2.5-7B, preferably on LoRA first.**
