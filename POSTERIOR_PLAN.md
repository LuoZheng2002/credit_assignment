# Posterior Implementation Plan

## Goal

Implement `src/direct_tool/direct_tree_posterior.rs` with MAP-style segment posterior fitting that follows `paper_draft/methodology.md`, while keeping behavior aligned with the optimization style used in `src/em/em_fitting.rs`.

## Core Formula to Match

For each judged leaf `l`:

- `y_l in {+1, -1}` from leaf correctness.
- `x_{l,i} in {0,1}` from whether segment `i` is on the leaf path.
- Segment parameters are `m_i` and `u_i` (`std_i = exp(u_i)`).

Leaf moments:

- `mu_l = sum_i x_{l,i} m_i`
- `var_l = sum_i x_{l,i} exp(2u_i)`
- `tau_l = sqrt(var_l + eps)`
- `z_l = y_l * mu_l / tau_l`

MAP objective:

- `J = -sum_l log Phi(z_l) + (1/(2*sigma_mean^2)) * sum_i (m_i - mu_0(T_i))^2 + (1/(2*sigma_log_std^2)) * sum_i u_i^2`

where `mu_0(T_i) = c * Phi^{-1}(clip(A(T_i), delta, 1-delta))`.

## Data Mapping from Direct Tree

1. Build a stable segment index from `DirectTree.segments`.
2. Build judged leaf observations from `leaf_segment_judgments`.
3. Recover each judged leaf path by walking `parent_id` up to root.
4. Convert correctness to `y_l` sign (`+1` or `-1`).
5. Use path membership as binary `x_{l,i}`.

## Temperature-Conditioned Prior Strategy

Given current repo data flow, external held-out calibration is not wired into `DirectTree` yet. Use an in-tree approximation:

1. Group judged leaves by their segment generation temperature.
2. Estimate bucket accuracy per temperature.
3. For segment temperature `T_i`, use nearest-bucket accuracy (fallback to global tree accuracy if needed).
4. Apply clipped probit transform to produce `mu_0(T_i)`.

This keeps the implementation consistent with the new methodology while avoiding additional schema/plumbing changes.

## Optimizer Structure

Follow the same overall style as `em_fitting.rs`:

1. Initialize `m_i` from `mu_0(T_i)` and `u_i = 0`.
2. Evaluate objective and analytic gradients.
3. Use gradient descent with backtracking line search.
4. Clamp `u_i` to a fixed range each step.
5. Stop when gradient norm is tiny, step search fails, or max iterations is reached.

## Numerical Stability

- Use stable `log Phi(z)` and `phi(z)/Phi(z)` evaluation in tails.
- Enforce finite positive `tau_l` and variance terms.
- Keep epsilon floor in denominator-like terms.
- Clip prior accuracy before inverse-CDF transform.

## Output Contract

Return `BTreeMap<SegmentId, Posterior>` where:

- `Posterior.mean = m_i as f32`
- `Posterior.log_std = u_i as f32`

These values are consumed by branching score conversion in `direct_tree_to_actions.rs`.
