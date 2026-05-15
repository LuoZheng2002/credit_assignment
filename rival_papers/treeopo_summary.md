# Tree-OPO Summary (Methodology + Implementation)

## Paper identity

- **Title:** Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning
- **Core goal:** Improve GRPO-style RL on reasoning tasks by reusing **offline MCTS teacher prefixes** and doing **prefix-aware advantage estimation**.

## Methodology

1. **Reframe GRPO into staged, tree-structured training**
   - Standard GRPO treats samples as coming from one flat prompt group.
   - Tree-OPO instead uses an offline teacher-generated MCTS tree of prefixes.
   - Training samples are mixed across different prefix depths (easy deep prefixes, harder shallow prefixes), yielding an implicit reverse curriculum.

2. **Off-policy prefix sampling + online student completion**
   - Offline: teacher (stronger model) runs multi-turn MCTS and stores prefix tree.
   - Online update: sample `K` prefixes from fixed dataset distribution `rho_D`.
   - Student policy completes each sampled prefix; terminal reward is binary (`0/1`).

3. **Main challenge identified**
   - Rewards from different prefixes have different expected returns `E[r | p]`.
   - Flat centering across mixed prefixes causes biased/misaligned credit assignment and higher gradient variance.

4. **Staged Advantage Estimation (SAE)**
   - Advantages are solved via constrained projection/QP to enforce:
     - zero-mean advantages,
     - norm control,
     - tree-consistent ordering constraints derived from parent/child and sibling relationships.
   - Two forms:
     - **Hard SAE (practical):** non-convex normalization (`||a||^2 = N`) + margins.
     - **Soft SAE (theory):** convex relaxation (`||a||^2 <= N`) + usually zero margins.

5. **Practical low-cost baseline estimators for `V(p)`**
   - Raw advantage: `a'_i = r_i - alpha * V(p_i)`, then mean-centered.
   - Heuristics for `V(p)`:
     - **Expectation / empirical success rate** over subtree,
     - **Optimistic:** any success in subtree,
     - **Pessimistic:** no failure in subtree.
   - Expectation heuristic is the preferred default in results.

6. **Optional off-policy correction**
   - If teacher policy density is available, uses GSPO-style importance ratio `pi_theta / pi_T` to correct sampling bias.

## Theoretical claims

- Policy gradient remains unbiased under prefix-dependent baselines.
- Variance-optimal deterministic baseline is `V(p) = E[r | p]`.
- SAE projection with tree constraints is claimed to reduce or preserve variance vs unstructured centered rewards.
- They recommend mean-centering without std-scaling to preserve stage-relative magnitude semantics.

## Implementation details (what to reproduce)

### Data and models

- **Training task:** GSM8K (plus GSM8K-MCTS prefix dataset).
- **Prefix source:** MCTS-generated traces, with all prefixes expanded into staged examples.
- **Scale in paper:** ~160k unique prefixes from rollout settings like 16 rollouts/problem, depth up to 5, branching up to 5.
- **Student:** Qwen2.5-1.5B.
- **Teacher (when applicable):** Qwen2.5-7B.

### Optimization setup (reported)

- Policy objective combines policy gradient term and optional reverse-KL distillation term.
- Key listed hyperparameters include:
  - `alpha = 0.5` (baseline weight in advantage),
  - group size `8`,
  - AdamW, cosine LR,
  - policy LR around `3e-5` (distillation component LR listed separately),
  - bf16,
  - top-p `1.0`, temperature `1.0`,
  - LoRA (`r=16`, alpha `64`, dropout `0.1`).
- Max lengths differ by setup:
  - GRPO: prompt 256 / sequence 512,
  - Tree-OPO: prompt 512 / sequence 768.

### SAE solver

- Implemented via constrained optimization (SLSQP in `scipy.optimize.minimize`).
- Warm-start from mean-centered rewards.
- Hard version used in current experiments; soft/penalized versions discussed as extensions.

### Experiment structure

- Compare advantage structures:
  - flat,
  - trace,
  - tree (hierarchical MCTS-prefix-aware).
- Evaluate pass@1 on GSM8K and cross-dataset checks (e.g., GSM-Symbolic, MATH).

## Reported empirical pattern

- Tree-structured expectation baseline outperforms flat/trace variants on GSM8K in their setup.
- Hard SAE can destabilize due to rigid norm constraint.
- Soft SAE performs close to expectation heuristic and is more stable than hard SAE.
- Gains are characterized as modest on GSM8K (possibly due to simpler tree structure).

## Practical takeaway

- Tree-OPO is best viewed as **offline MCTS-prefix curriculum + prefix-aware advantage shaping** for GRPO-like training.
- The key engineering value is not adding a learned critic, but injecting tree consistency into advantages with lightweight heuristics or constrained projection.
