# TreeRPO Paper Weakness Summary

## Main weakness themes from reviews

1. **High compute overhead from tree sampling**
   - Reviewers repeatedly flagged that TreeRPO may be expensive despite being reward-model-free.
   - Main concern: dense step-level signal is obtained via substantial rollout/sampling cost, potentially hurting practicality vs simpler baselines.
   - Requested clarification: GPU-hour overhead relative to GRPO under matched sample budgets.

2. **Baseline coverage was incomplete**
   - Reviewers asked for comparisons against PRM-based and step-level methods (e.g., Math-Shepherd, Step-DPO), not only GRPO.
   - Some also questioned novelty claims without stronger head-to-head evidence versus closely related contemporary tree-credit methods.

3. **Ablation and sensitivity analysis were insufficient**
   - Key hyperparameters were mostly fixed (tree depth, branching factor, pruning threshold) without full sweeps.
   - Reviewers requested stronger isolation of contributing factors, especially:
     - tree sampling itself vs objective-side additions (e.g., KL term),
     - effect of advantage renormalization,
     - controls such as `D=1` and `N=1` under fixed sample budgets.

4. **Limited statistical rigor in reported results**
   - Results were criticized for relying on single-run curves without uncertainty estimates (std/confidence intervals).
   - Concern: benchmark sample sizes are small enough that variance may materially affect conclusions.

5. **Questions on scale, generality, and robustness**
   - Experiments centered on Qwen2.5-Math (mainly 1.5B; limited 7B analysis), and gains appeared smaller at larger scale.
   - Reviewers asked whether benefits persist on larger models and non-math domains.
   - Some questioned whether gains remain when output length is controlled, to rule out confounding from shorter responses.

6. **Theoretical and methodological justification gaps**
   - Reviewers asked for stronger theory/intuition on why larger branching improves optimization and how estimator variance compares to GRPO.
   - Step segmentation by fixed token length (`L_step=384`) was seen as potentially misaligned with semantic reasoning steps.
   - Some implementation details (verification function parity between TreeRPO and GRPO) were requested for reproducibility.

## Overall reviewer risk assessment

- Reviewers generally agreed the direction is promising and practically relevant, but viewed evidence as not yet strong enough for broad claims.
- The biggest acceptance risk was a combination of **compute-cost concerns**, **insufficient ablations/statistical evidence**, and **uncertain scalability/generality** beyond the current math-focused setup.
