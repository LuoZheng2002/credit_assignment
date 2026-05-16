# MAP Credit Assignment

This document defines a maximum-a-posteriori (MAP) framework for segment-level credit assignment in trajectory trees. The objective is to infer signed segment contributions and uncertainty from judged rollout outcomes, while incorporating a temperature-conditioned prior mean.

## Problem Setup

For each judged leaf trajectory `l`:

- `y_l in {+1, -1}` denotes correctness (+1 correct, -1 incorrect).
- `x_{l,i} in {0,1}` indicates whether segment `i` lies on leaf path `l`.
- Segment parameters are:
  - mean contribution `m_i`
  - log standard deviation `u_i`, with `std_i = exp(u_i)` and `var_i = exp(2u_i)`
  - temperature tag `T_i` (the decoding temperature under which the segment is generated)

Leaf-level moments:

- `mu_l = sum_i x_{l,i} m_i`
- `var_l = sum_i x_{l,i} exp(2u_i)`
- `tau_l = sqrt(var_l + eps)`

Leaf normalized margin:

- `z_l = y_l * mu_l / tau_l`

Likelihood:

- `p(y_l | params) = Phi(z_l)` where `Phi` is the standard normal CDF.

## Temperature-Conditioned Prior Mean

Let `A(T)` denote model accuracy on a held-out evaluation set under decoding temperature `T`.

This prior calibration is an **Empirical Bayes** procedure (Type-II Empirical Bayes): the prior hyperparameter `mu_0(T)` is estimated from evaluation data rather than fixed a priori.

We convert `A(T)` into a prior center by a probit mapping:

- `mu_0(T) = c * Phi^{-1}(clip(A(T), delta, 1-delta))`

where `c > 0` is a scale factor and `delta` is a small clipping constant for numerical stability.

Interpretation:

- If accuracy at temperature `T` is above chance, `mu_0(T)` is positive.
- If it is below chance, `mu_0(T)` is negative.
- Magnitude reflects confidence implied by the observed accuracy.

Each segment prior center is then temperature-specific:

- `m_i ~ N(mu_0(T_i), sigma_mean^2)`

## MAP Objective

We minimize the negative log posterior:

`J = J_likelihood + J_mean_prior + J_log_std_prior`

with:

- `J_likelihood = -sum_l log Phi(z_l)`
- `J_mean_prior = (1 / (2 * sigma_mean^2)) * sum_i (m_i - mu_0(T_i))^2`
- `J_log_std_prior = (1 / (2 * sigma_log_std^2)) * sum_i u_i^2`

Key properties:

- The probit likelihood provides directional gradients from initialization.
- Margin normalization by `tau_l` couples credit assignment with uncertainty.
- Temperature-conditioned priors inject externally measured decoding behavior without introducing node-type heuristics.

## Numerical and Optimization Details

Optimization and stability practices:

- clamp `u_i` to a fixed range (e.g., `[-4, 2]`)
- enforce finite positive hyperparameters (`sigma_mean`, `sigma_log_std`, `eps`)
- use numerically stable `log Phi` computation
- optimize with gradient methods and backtracking line search

Diagnostics remain the same:

- objective trace
- train sign accuracy
- per-leaf normalized margins

## Practical Calibration Procedure

1. Select a temperature grid `T in {T_1, ..., T_K}`.
2. For each `T_k`, evaluate the base model on a fixed evaluation set and compute `A(T_k)`.
3. Compute `mu_0(T_k)` from the probit mapping.
4. Assign each segment its associated `T_i` and look up/interpolate `mu_0(T_i)` during fitting.

If a segment temperature is not exactly on the grid, use interpolation or nearest-neighbor lookup as a deterministic preprocessing rule.
