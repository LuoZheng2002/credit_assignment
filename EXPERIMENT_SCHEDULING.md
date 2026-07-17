# Experiment and Scheduling Plan

Last updated: 2026-07-15

## Purpose

This document defines the near-term experimental plan for the TreeMAPPO paper under the current practical constraint:

- We are **not** yet ready to spend production compute on the full `bin_orchestrator.rs` pipeline.
- We are still in the **hyperparameter stabilization / systems validation** phase.
- Therefore, the immediate workflow should prefer the split pipeline:
  1. `oneshot_rollout`
  2. `oneshot_generation`
  3. `oneshot_training`

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
- During hyperparameter search, prioritize LoRA single-GPU training over full-parameter FSDP training unless a specific full-model question is being answered.
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

4. **Generation should remain sorted by ascending length during hyperparameter search.**
   - `training_set_sort_mode = "ByLengthAscending"` is already enabled in the no-tool GRPO generation config.
   - This is the right default during stabilization because it exposes a usable non-OOM prefix early.

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

## Running / Queued Experiments

### 1. Full-parameter FSDP smoke test

- Job: `20217704`
- Status: `PENDING`
- Pending reason: `Priority`
- Config intent:
  - `qwen25`
  - no-tool
  - GRPO-generated dataset
  - full-parameter training
  - `fsdp`
  - `num_gpus = 4`
  - smoke time budget

What this run is for:

- Validate full-parameter 4-GPU training at the current memory footprint.
- Determine whether the current `training_trajectory_len_cutoff = 4096` is too high in practice.
- If it OOMs, extract `longest_non_oom_trajectory_length` and use that to set the next cutoff.

Interpretation rule:

- If it succeeds: promote full FSDP no-tool GRPO to a real run.
- If it OOMs after some progress: lower cutoff to the reported longest non-OOM length and rerun.
- If it OOMs before any sample trains: treat the config as invalid and lower the cutoff more aggressively.

### 2. Background CPU job unrelated to the core paper schedule

- Job: `20219942`
- Name: `company-psro-cpu`
- Status: `RUNNING`

Implication:

- This is not part of the TreeMAPPO experiment matrix.
- It may still consume queue attention or user time, so it should be treated as background contention when planning CPU jobs.

## Near-Term Experimental Strategy

We should use a staged adaptive plan instead of launching the full paper sweep immediately.

### Phase 0: Stabilize the training recipe

Goal:

- Identify one robust no-tool GRPO training recipe for:
  - LoRA single-GPU (primary path)
  - full-parameter FSDP 4-GPU (secondary path)

Promotion criteria:

- At least one successful serious run for each training regime.
- No resource-allocation bug.
- No distributed OOM wraparound.
- A known usable trajectory cutoff for each regime.

This phase is the current priority. Within this phase, LoRA should be optimized first because it is faster to queue, cheaper to iterate, and already has one successful serious run.

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

### If the full FSDP smoke run succeeds

Then queue, in order:

1. **Serious full no-tool GRPO run**
   - same config family as smoke
   - normal time budget
2. **No-tool TreeMAPPO generation prep**
   - CPU-only
3. **No-tool TreeMAPPO training**
   - LoRA first if tree rollout/generation is already ready
   - otherwise full only after generation completes

Scientific effect:

- This immediately gives a GRPO full-vs-LoRA comparison and starts the first TreeMAPPO no-tool main comparison.

### If the full FSDP smoke run OOMs after partial progress

Then:

1. Read `training_summary.json`
2. Set the next full-training cutoff to `longest_non_oom_trajectory_length`
3. Rerun the smoke test once
4. If the rerun succeeds, promote to serious full training

Do not:

- launch multiple full-FSDP serious jobs before the cutoff is validated.

### If the full FSDP smoke run OOMs before any sample trains

Then:

1. Treat the current full-training cutoff as invalid.
2. Lower the cutoff materially below the attempted region.
3. Prefer a shorter smoke rerun before spending more queue time.

### If LoRA and full both become stable

Then prioritize:

1. GRPO no-tool LoRA
2. TreeMAPPO no-tool LoRA
3. LoRA rank sweep for the best no-tool condition
4. GRPO no-tool full
5. TreeMAPPO no-tool full

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

### Slot 1: Keep current pending full FSDP smoke as a secondary path

- job `20217704`

No action unless:

- it OOMs,
- it is blocked too long and we decide to replace it with a LoRA-priority job,
- we need its result to answer a specific full-model question.

### Slot 2: Prioritize LoRA no-tool follow-ups

Action:

- queue additional single-GPU LoRA experiments before launching more 4-GPU full runs.
- the first preferred branch is a LoRA rank sweep around the already-successful no-tool GRPO setup.

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

### Slot 3: Prepare TreeMAPPO no-tool generation artifacts on CPU

Action:

- launch / verify the no-tool TreeMAPPO generation pipeline using the already valid non-LoRA rollout prerequisites.

Reason:

- This keeps the next scientifically useful comparison ready while the FSDP smoke waits in queue.

### Slot 4: Prepare evaluation / summary scripts for the first main table row

Action:

- standardize extraction of:
  - baseline accuracy
  - GRPO no-tool accuracy
  - TreeMAPPO no-tool accuracy
  - per-dataset breakdown

Reason:

- Results without immediate aggregation slow down decision-making.

### Slot 5: Queue one follow-up training job only when justified by Slot 1

Branch:

- If `20217704` succeeds:
  - queue serious full no-tool GRPO
- If `20217704` OOMs with progress:
  - queue reduced-cutoff rerun
- If `20217704` OOMs immediately:
  - queue a more conservative smoke rerun

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
4. **Full-parameter no-tool training stability is secondary to LoRA iteration speed.**
5. **We should not run orchestrator-scale jobs until the LoRA recipe is stabilized.**
6. **The next paper-central milestone is the first valid no-tool GRPO vs TreeMAPPO comparison on Qwen2.5-7B, preferably on LoRA first.**
