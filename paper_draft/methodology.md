# Methodology

This project studies fine-grained credit assignment for agentic LLM training through token-level trajectory branching. We introduce **TreeMAPPO** (Tree-based Maximum-A-Posteriori Policy Optimization), a reference-free optimization framework that estimates segment-level contribution and uncertainty within a rollout tree for fine-grained credit assignment. More broadly, this formulation converts sparse trajectory-level outcomes into dense, uncertainty-aware training signals over reasoning paths, enabling more stable and data-efficient optimization of agent behavior without requiring reference trajectories or step-level supervision.

## Agent Interaction Protocol

Our experimental setup uses a lightweight agent framework with one callable Python tool as the intended interaction protocol for both training and evaluation.

1. Each prompt contains the problem statement and the tool specification.
2. The model is allowed to interleave reasoning and tool use.
3. Tool calls are delimited by `<tool_wait></tool_wait>`. Once `</tool_wait>` is emitted, generation is paused, the tool is executed, and the tool response is injected into the assistant context.
4. The model is instructed to present the final answer in `\boxed{}`. Generation stops when a boxed answer is detected at a control handoff (EOS or `</tool_wait>`).
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

### Adaptive Branching Temperature

When creating a new branch, we adapt decoding temperature based on the current empirical tree accuracy.

- Let `acc_tree` be the fraction of judged leaves in the current tree that are correct.
- Let `T_prev` be the decoding temperature used by the most recently completed branch.
- If `acc_tree > 0.65` (overly favorable regime), increase temperature to encourage more challenging and diverse continuations.
- If `acc_tree < 0.35` (overly unfavorable regime), decrease temperature to improve branch solvability.
- Otherwise, keep temperature near `T_prev`.

This policy counteracts accuracy saturation at either extreme and improves information gain from new branches.

Because segment prior means are calibrated as a function of temperature (Empirical Bayes calibration), segments generated at adjusted temperatures receive temperature-consistent priors. In principle, this reduces potential bias propagation to ancestor segments and preserves fairness in credit assignment across mixed-temperature branches.

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
- Each segment `i` has parameters `m_i` (mean contribution), `u_i` (log standard deviation), and `T_i` (decoding temperature tag).

Leaf-level moments are:

- `mu_l = sum_i x_{l,i} m_i`
- `var_l = sum_i x_{l,i} exp(2u_i)`
- `tau_l = sqrt(var_l + eps)`

with normalized signed margin:

- `z_l = y_l * mu_l / tau_l`

and probit likelihood:

- `p(y_l | params) = Phi(z_l)` where `Phi` is the standard normal CDF.

### Temperature-Conditioned Prior (Empirical Bayes)

Let `A(T)` be model accuracy on a held-out evaluation set at decoding temperature `T`. We calibrate prior mean as:

- `mu_0(T) = c * Phi^{-1}(clip(A(T), delta, 1-delta))`

where `c > 0` is a scale factor and `delta` is a small clipping constant.

This is an **Empirical Bayes** (Type-II Empirical Bayes) calibration procedure: the prior hyperparameter `mu_0(T)` is estimated from data.

Each segment prior is temperature-dependent:

- `m_i ~ N(mu_0(T_i), sigma_mean^2)`

Interpretation:

- `A(T) > 0.5` implies positive prior center.
- `A(T) < 0.5` implies negative prior center.
- Larger distance from `0.5` yields larger `|mu_0(T)|`.

### MAP Objective

We minimize negative log posterior:

- `J = J_likelihood + J_mean_prior + J_log_std_prior`

with:

- `J_likelihood = -sum_l log Phi(z_l)`
- `J_mean_prior = (1 / (2 * sigma_mean^2)) * sum_i (m_i - mu_0(T_i))^2`
- `J_log_std_prior = (1 / (2 * sigma_log_std^2)) * sum_i u_i^2`

This objective encourages the signed trajectory contribution sum (`mu_l`) to move away from zero in the direction implied by outcome label `y_l`, while uncertainty regularization prevents degenerate variance inflation.

### Optimization and Calibration Procedure

Numerical and optimization settings:

- clamp `u_i` to a fixed range (e.g., `[-4, 2]`)
- enforce finite positive `sigma_mean`, `sigma_log_std`, and `eps`
- use numerically stable `log Phi`
- optimize with gradient methods and backtracking line search

Calibration workflow:

1. Choose temperature grid `T in {T_1, ..., T_K}`.
2. Evaluate base-model accuracy `A(T_k)` on a fixed held-out set for each `T_k`.
3. Compute `mu_0(T_k)` from the probit mapping.
4. For each segment, use lookup/interpolation to obtain `mu_0(T_i)`.

### Branching Scores from Segment Estimates

For branch-point selection, we map posterior mean and variance to a bounded branching score in `[0, 1]`, where values closer to `1` are prioritized.

Given segment posterior moments `m_i` and `u_i` (`var_i = exp(2u_i)`), we first normalize statistics across all segments in the tree:

```text
m_i_norm = (m_i - mean(m)) / (std(m) + eps)
log_var_i = ln(var_i + eps)
log_var_i_norm = (log_var_i - mean(log_var)) / (std(log_var) + eps)
var_i_norm = exp(log_var_i_norm)
```

Define the uncertainty-adjusted ratio:

```text
ratio_i = |m_i_norm| / sqrt(var_i_norm + eps)
```

and the segment branching score:

```text
segment_branch_score_i = exp(-alpha * ratio_i^2),   alpha > 0
```

Properties:

- `segment_branch_score_i in (0, 1]`
- `ratio_i -> 0` implies score `-> 1` (highest branching priority)
- larger `ratio_i` implies lower score

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

After tree construction, each segment advantage is set to its signed MAP contribution estimate. Advantages are normalized but not mean-shifted.

## Training Set Construction

Trees are flattened into trajectories that may share prefixes. To avoid repeatedly training shared segments, each segment is assigned to at most one training trajectory.

We use a greedy selection policy:

1. Select the trajectory with the highest per-token average absolute advantage.
2. Mark all of its segments as consumed.
3. Repeat selection among trajectories that still contain unconsumed, learnable segments.
4. Aggregate selected trajectories across trees, rank by average absolute advantage, and prune the lowest-contribution tail.

## Training Sample Ordering

Selected trajectories are ordered by sequence length to improve batch efficiency. Since low-value trajectories are already pruned during selection, no additional value-based ordering is applied.

## Training Interface

Each training sample provides:

1. `input_ids`: token IDs.
2. `labels`: token IDs for trainable positions and `-100` for masked positions; the first tool-response token may use EOS depending on generation termination handling in vLLM.
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

Out-of-distribution datasets:

- AIME25
- AMC23

## Experiments

### Main Evaluation

1. Measure baseline pass@1 accuracy for each model on all six evaluation datasets.
2. Train TreeMAPPO on three training sets (one per in-distribution dataset) for each model.
3. Re-evaluate pass@1 on all six datasets.
4. Run five independent trials and report confidence intervals.

### Ablation Studies

1. **Fine-grained credit assignment (vs. GRPO):** replace segment-level scoring with group-relative rollout advantage.
2. **Branch-point guidance (vs. TEMPO-style natural divergence):** remove explicit branch-point selection and rely on spontaneous divergence across rollouts.
3. **MAP credit model (vs. TreeRPO):** replace MAP-based segment scoring with leaf-dominant heuristic advantage assignment.
4. **Optional baseline (TreePO):** compare against subgroup-relative segment advantages among shared-prefix trajectories.
5. **Branching budget sensitivity:** vary the number of branches per tree (e.g., low/medium/high expansion budgets) to quantify the trade-off between sample efficiency, segment utilization, and downstream pass@1.

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
