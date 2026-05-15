# TEMPO Paper Weakness Summary

## Main weakness themes from reviews

1. **Limited branch overlap may cap token-level credit gains**
   - Multiple reviewers argued that sampled trajectories quickly diverge, so shared prefixes are mostly early tokens.
   - As a result, branch-based TD credit may only apply to a small subset of tokens, with much of training effectively reverting to GRPO.
   - Concern: this limits scalability to long-horizon reasoning where late-token credit assignment matters most.

2. **Questions about long-context realism and setup clarity**
   - Reviewers initially interpreted the setup as using short max lengths (around 1k), and argued this might not stress true long-horizon delayed reward.
   - Related concern: if sequence lengths are large (8k-32k), branch sharing may become even sparser, further reducing TEMPO’s advantage.
   - There was also confusion about evaluation settings (e.g., thinking mode, MATH vs. MATH500), which weakened confidence in claims.

3. **Evaluation breadth and convergence evidence were seen as insufficient**
   - Reviewers requested stronger evidence that training fully converges (e.g., more iterations / stronger justification for stopping early).
   - Some felt the empirical results did not yet fully support the paper’s major hypothesis under broader conditions.

4. **Novelty concerns versus prior tree-based RL methods (especially TreeRPO)**
   - A recurring criticism was that the contribution appears incremental relative to existing tree-based approaches.
   - The lack of direct TreeRPO comparison in the initial submission was viewed as a key gap.

5. **Method-design questions and missing justifications**
   - Reviewers asked for clearer explanation of branching criterion/timing and whether branch count should be adaptive.
   - They also questioned certain design choices in advantage normalization (e.g., scaling TD by `std(r)` instead of TD variance), asking for stronger theoretical or empirical justification.

## Overall reviewer risk assessment

- The central idea was generally considered intuitive and useful, but several reviewers doubted its **magnitude of impact** in realistic long-chain settings.
- Main perceived risk: TEMPO may provide meaningful but **bounded** improvements (a practical add-on to GRPO), rather than a substantial step-change in credit assignment.
