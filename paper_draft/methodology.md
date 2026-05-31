# Methodology

This project studies fine-grained credit assignment for agentic LLM training through token-level trajectory branching. We introduce **TreeMAPPO** (Tree-based Maximum-A-Posteriori Policy Optimization), a reference-free optimization framework that estimates segment-level contribution and uncertainty within a rollout tree for fine-grained credit assignment. More broadly, this formulation converts sparse trajectory-level outcomes into dense, uncertainty-aware training signals over reasoning paths, enabling more stable and data-efficient optimization of agent behavior without requiring reference trajectories or step-level supervision.

## Agent Interaction Protocol

Our experimental setup uses a lightweight agent framework with one callable Python tool as the intended interaction protocol for both training and evaluation.

1. Each prompt contains the problem statement and the tool specification.
2. The model is allowed to interleave reasoning and tool use.
3. Tool calls are delimited by fenced Python markdown blocks. Once the closing fence (` ``` ` on its own line) is emitted, generation is paused, the tool is executed, and the tool response is injected into the assistant context.
4. The model is instructed to present the final answer in `\boxed{}`. Generation stops when a boxed answer is detected at a control handoff (EOS or tool-call fence stop).
5. If EOS is reached without a boxed answer, the framework appends a continuation cue requesting either further reasoning or a boxed final answer.

## Trajectory Branching

For each question, we first sample 4 complete trajectories. We then iteratively expand a trajectory tree by branching from selected token positions.

- Branching is token-level (not restricted to sentence or step boundaries).
- After selecting a branching position, the model re-generates from that position to completion to form a new branch.
- Candidate branch points are of two types:
  1. **Segment split:** split an existing segment (e.g., at its midpoint).
  2. **Node expansion:** branch again from an existing branching node, increasing its branching factor.

### Branch-Point Selection Criteria

Branch selection is guided by a score with regularization terms:

1. **Relative length penalty (segment split only):** short segments receive higher penalty to avoid over-fragmentation.
2. **Branching-factor penalty (node expansion only):** high-degree nodes receive higher penalty to prevent over-concentration.
3. **Segment or node score:** uncertainty-aware contribution estimate (defined below).

Both penalties are implemented as multiplicative factors `< 1.0`:

```text
segment_multiplier_i = 1 - exp(-k_seg * (len_i - 1))
node_multiplier_n = exp(-k_node * (b_n - 1))
```

where `len_i in [1, +inf)` is reasoning-token length for segment `i`, `b_n in [1, +inf)` is node branching factor, and `k_seg, k_node > 0` control falloff strength.

These penalties encourage balanced exploration across the full tree.

### Decoding Temperature Policy

In the current implementation, decoding temperature is fixed within a rollout tree.

- A single value `T_fixed` is specified in rollout config and reused for prompt continuation, trunk trajectories, and all branch expansions.
- We do not currently apply adaptive per-branch temperature updates.

This keeps tree construction and posterior estimation behavior consistent across all segments in a tree.

## Segment and Node Scoring

### Intuition for the Contribution Model

The MAP model adopts a simplified but useful assumption: each segment contributes a signed quantity to the final outcome, and trajectory correctness is determined by the sign of the summed contribution. In this view, errors can be corrected by later reasoning, but repeated errors generally accumulate negative evidence and reduce the probability of a correct final answer.

This assumption yields several desirable credit-assignment behaviors:

1. If a segment is followed by a correct leaf, that segment is likely helpful, but not necessarily decisive.
2. If two sibling continuations disagree in final correctness, the split segment is inferred with higher confidence as a key positive or negative contributor, while ancestor contributions are correspondingly softened and less certain.
3. For early segments with many descendants, mixed descendant outcomes imply uncertainty in that early segment's sign.
4. If descendants are predominantly correct, the early segment can receive both high confidence and large positive contribution; in some cases its estimated advantage can exceed that of individual leaf descendants because it is supported by broader subtree evidence.

This behavior is not captured by simple heuristics (e.g., local win-loss counting), which cannot represent uncertainty and hierarchical evidence propagation in the same principled way. We now formalize this intuition as a probabilistic MAP model.

### MAP Credit Assignment Model

For each judged leaf trajectory `l`:

- `y_l in {+1, -1}` denotes correctness (+1 correct, -1 incorrect).
- `x_{l,i} in {0,1}` indicates whether segment `i` is on the path to leaf `l`.
- Each segment `i` has parameters `m_i` (mean contribution) and `u_i` (log standard deviation).

Leaf-level moments are:

- `mu_l = sum_i x_{l,i} m_i`
- `var_l = sum_i x_{l,i} exp(2u_i)`
- `tau_l = sqrt(var_l + eps)`

with normalized signed margin:

- `z_l = y_l * mu_l / tau_l`

and probit likelihood:

- `p(y_l | params) = Phi(z_l)` where `Phi` is the standard normal CDF.

### MAP Objective

We minimize negative log posterior:

- `J = J_likelihood + J_mean_prior + J_log_std_prior`

with:

- `J_likelihood = -sum_l log Phi(z_l)`
- `J_mean_prior = (1 / (2 * sigma_mean^2)) * sum_i m_i^2`
- `J_log_std_prior = (1 / (2 * sigma_log_std^2)) * sum_i u_i^2`

This objective encourages the signed trajectory contribution sum (`mu_l`) to move away from zero in the direction implied by outcome label `y_l`, while uncertainty regularization prevents degenerate variance inflation.

### Optimization and Calibration Procedure

Numerical and optimization settings:

- clamp `u_i` to a fixed range (e.g., `[-4, 2]`)
- enforce finite positive `sigma_mean`, `sigma_log_std`, and `eps`
- use numerically stable `log Phi`
- optimize with gradient methods and backtracking line search

Calibration workflow:

1. Choose a rollout temperature `T_fixed` for tree generation.
2. Posterior means are initialized at zero (neutral prior center).

### Branching Scores from Segment Estimates

For branch-point selection, we map posterior mean and standard deviation to a bounded branching score in `[0, 1]`, where values closer to `1` are prioritized.

Given segment posterior moments `m_i` and `u_i` (`std_i = exp(u_i)`), we first compute:

```text
r_i = m_i / (std_i + eps)
```

Then we normalize `r_i` across all segments in the tree:

```text
r_i_norm = (r_i - mean(r)) / (std(r) + eps)
```

and define the segment branching score:

```text
segment_branch_score_i = exp(-alpha * r_i_norm^2),   alpha > 0
```

Properties:

- `segment_branch_score_i in (0, 1]`
- `ratio_i -> 0` implies score `-> 1` (highest branching priority)
- larger `|r_i_norm|` implies lower score

Node score is derived from incident segment branching scores. For a node with branching factor `b`, parent segment score `s_p`, and child segment scores `s_{c_i}`:

```text
node_score = (b * s_p + sum_{i=1..b} s_{c_i}) / (2b)
```

This weighting gives equal total mass to parent-side and child-side evidence.

Final branch-point priorities are:

```text
segment_priority_i = segment_branch_score_i * segment_multiplier_i
node_priority_n = node_score_n * node_multiplier_n
```

## Advantage Assignment

After tree construction, each segment starts with an unnormalized signed MAP contribution estimate:

```text
r_i = m_i / std_i
```

To reduce instability from very short segments, we length-scale this value by reasoning-token length with tree-level clamping:

```text
L_max = max_j reasoning_only_token_length_j
L_floor = max(1, L_max / 8)
L_i_clamped = max(reasoning_only_token_length_i, L_floor)
r_i_len = r_i / L_i_clamped
```

We then normalize `r_i_len` across segments in the same tree using standard deviation scaling (without mean-shift sign flips).

## Training Set Construction

Trees are flattened into trajectories that may share prefixes. To avoid repeatedly training shared segments, each segment is assigned to at most one training trajectory.

We use a greedy selection policy:

1. Select the trajectory with the highest per-token average absolute advantage.
2. Mark all of its segments as consumed.
3. Repeat selection among trajectories that still contain unconsumed, learnable segments.
4. Aggregate selected trajectories across trees, rank by average absolute advantage, and prune the lowest-contribution tail.

## Training Sample Ordering

Selected trajectories are ordered by sequence length to improve batch efficiency. Since low-value trajectories are already pruned during selection, no additional value-based ordering is applied.

## Epoch Definition in Training Pipeline

We define **one epoch as one training program run** over an epoch-specific training dataset.

- Within a run, `num_iterations` controls repeated passes over that epoch's dataset; these are optimization iterations, not epochs.
- Across epochs, the dataset is regenerated using the model produced by the previous epoch.
- Checkpointing is scoped to a single run/epoch directory (`checkpoints_parent_dir/checkpoints`), and checkpoint internals are treated as implementation-private resume state.

## Training Interface

Each training sample provides:

1. `input_ids`: token IDs.
2. `labels`: token IDs for trainable positions and `-100` for masked positions; the first tool-response token may use EOS depending on generation termination handling in SgLang.
3. `attention_mask`: padding mask.
4. `advantages`: normalized token-level advantage values aligned with `input_ids`.

## Implementation Details

### Models

1. `Qwen/Qwen2.5-7B-Instruct`
2. `Qwen/Qwen3-4B` (thinking mode enabled)
3. `Qwen/Qwen3.5-4B` (thinking mode enabled)

### Compute and Optimization Setup

- Hardware: 4 x A100 GPUs.
- Primary training plan: LoRA (PEFT) with DDP.
- Backup plan: full-parameter training with FSDP.

### Data Sources

In-distribution datasets:

- DeepMath
- MATH
- GSM8K

In-distribution train/validation construction:

- We construct both `hybrid_train.sqlite` and `hybrid_val.sqlite` from the train splits of DeepMath, MATH, and GSM8K.
- Per dataset, training uses 5,000 samples and validation uses 1,000 samples.
- Validation samples are explicitly non-overlapping with training samples by using disjoint index ranges (`question_id` 0-4999 for train and 5000-5999 for validation).
- We enforce a dataset-size assertion before extraction: each in-distribution dataset must have at least 6,000 train rows so both splits can be formed without overlap.

Out-of-distribution datasets:

- AIME25
- AMC23

Test set construction:

- We construct `aggregated_test.sqlite` by concatenating samples from all five evaluation datasets in the following order: DeepMath, MATH, GSM8K, AIME25, AMC23.
- Per dataset, up to 1,000 samples are drawn from the test split where available (MATH, GSM8K, AIME25, AMC23). For DeepMath, which lacks a dedicated test split, samples are drawn from the train split starting after the combined train+val clearance of 6,000 rows (indices 0-5999 reserved for hybrid_train and hybrid_val), ensuring no overlap with the in-distribution training or validation sets.
- Samples from datasets with more than 1,000 rows are randomly subsampled with a fixed seed (42). Datasets with fewer than 1,000 rows use all available rows without repetition.
- Total test set size is approximately 5,000 rows (1,000 per dataset when source sizes permit).

## Experiments

### Main Evaluation

1. Measure baseline pass@1 accuracy for each model on all five evaluation datasets.
2. Train TreeMAPPO on the in-distribution training data (`hybrid_train.sqlite`) for each model.
3. Use in-distribution validation data (`hybrid_val.sqlite`) for checkpoint selection and hyperparameter control.
4. Re-evaluate pass@1 on all five datasets.
5. Run five independent trials and report confidence intervals.

### Ablation Studies

1. **Fine-grained credit assignment (vs. GRPO):** replace segment-level scoring with group-relative rollout advantage.
2. **Branch-point guidance (vs. TEMPO-style natural divergence):** remove explicit branch-point selection and rely on spontaneous divergence across rollouts.
3. **MAP credit model (vs. TreeRPO):** replace MAP-based segment scoring with leaf-dominant heuristic advantage assignment.
4. **Optional baseline (TreePO):** compare against subgroup-relative segment advantages among shared-prefix trajectories.
5. **Branching budget sensitivity:** vary the number of branches per tree (e.g., low/medium/high expansion budgets) to quantify the trade-off between sample efficiency, segment utilization, and downstream pass@1.
6. **Whether uses tools**

## Experiment Details

### Reporting Protocol

- Primary metric: pass@1 accuracy.
- Secondary reporting: confidence intervals from five runs.
- Evaluation scope: both in-distribution and out-of-distribution benchmarks.

### Fairness and Control Variables

- Use identical evaluation datasets and decoding settings across methods.
- Match model backbone and training budget where baseline implementations permit.
- Keep tool-calling environment and answer extraction (`\boxed{}`) consistent across all comparisons.

### Baseline Positioning

- **TEMPO:** comparison for natural vs. guided branching.
- **TreeRPO:** comparison for heuristic vs. probabilistic credit assignment.
- **TreePO (optional):** comparison for subgroup-relative segment credit.
- **GRPO:** comparison for trajectory/group-level rather than fine-grained segment-level advantage.

### Training Hyperparameters

- **Branches per tree:** TBD
- **Training dataset size and num epochs:** TBD
