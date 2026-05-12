## Goal

Estimate per-node contribution parameters from judged leaf outcomes with a probabilistic objective that avoids the all-zero fixed point.

This document replaces the previous EM-oriented plan with a MAP fitting plan.

---

## Why switch from EM framing to MAP

- Current implementation optimizes parameters directly with gradient methods.
- Deterministic sign + slack objective allows a stable all-zero stationary point.
- A probabilistic MAP objective provides directional gradients at initialization and encourages margin growth in the correct direction.
- We can keep the same data extraction pipeline and nearly all parameterization.

---

## Model and parameters

For each judged leaf `l`:

- Label: `y_l in {+1, -1}` (`+1` correct, `-1` incorrect)
- Path indicator: `x_{l,i} in {0,1}`
- Node contribution mean: `m_i`
- Node log-std: `u_i`, with `std_i = exp(u_i)` and `var_i = exp(2u_i)`

Leaf score moments:

- `mu_l = sum_i x_{l,i} m_i`
- `var_l = sum_i x_{l,i} exp(2u_i)`
- `tau_l = sqrt(var_l + eps)`

Node-type prior center:

- Ordinary (`NodeType::VerifierOff`): prior mean `ordinary_prior_mean` (data-calibrated from aggregated leaf accuracy)
- Special types: prior mean `ordinary_prior_mean + mu_k` (learned shared offset per special node type)

---

## MAP objective (chosen)

Use a probit likelihood on normalized leaf margin:

- `z_l = y_l * mu_l / tau_l`
- `p(y_l | params) = Phi(z_l)` where `Phi` is standard normal CDF

Minimize negative log posterior:

`J = J_likelihood + J_mean_prior + J_special_prior + J_log_std_prior`

where:

- `J_likelihood = -sum_l log Phi(z_l)`
- `J_mean_prior = (1 / (2 * sigma_ordinary^2)) * sum_i (m_i - mu_{node_type(i)})^2`
- `J_special_prior = (1 / (2 * sigma_special^2)) * sum_k mu_k^2`
- `J_log_std_prior = (1 / (2 * sigma_log_std^2)) * sum_i u_i^2`

This objective encourages:

- successful leaves to move toward larger positive normalized margin,
- failed leaves to move toward larger negative normalized margin,
- uncertainty-aware tradeoffs through `tau_l`.

---

## Numerical safeguards

- Clamp `u_i` to `[-4.0, 2.0]`
- Require finite positive sigmas: `sigma_ordinary`, `sigma_special`, `sigma_log_std`
- Require `eps >= 1e-8`
- Require non-empty judged leaves and valid root-to-leaf paths

For stable likelihood evaluation:

- Use a numerically safe implementation for `log Phi(z)`
- Use stable derivative computations in extreme tails

---

## Optimization strategy

- Gradient-based optimization with backtracking line search
- Keep current projected updates (`u_i` clamp)
- Track diagnostics:
  - objective trace
  - train sign accuracy
  - per-leaf normalized margins
  - optional top hard leaves by smallest `z_l`

Stopping:

- fixed `max_iterations` (current)
- optional future: gradient-norm tolerance

---

## Hyperparameter defaults (current)

- `sigma_ordinary = 1.0`
- `sigma_special = 1.0`
- `sigma_log_std = 1.0`
- `eps = 1e-6`
- `max_iterations = 100`
- `log_std clamp = [-4.0, 2.0]`

`lambda_slack` is removed from the new primary objective.

---

## Data boundary and artifacts

Input boundary stays the same:

- `EmDatasetBuilder::build_from_trees(&[Tree]) -> EmFitDataset`

Persisted fitting result stays compatible with current structs:

- per-node: `mean`, `log_std`, `node_type`, id bindings
- global: special node-type prior offsets `mu_k` relative to ordinary baseline
- diagnostics: objective trace, convergence flag, train metrics
- meta/config: includes `ordinary_prior_mean`

---

## Change log

- Ordinary prior mean is no longer fixed at `0`; it is calibrated from aggregated judged-leaf accuracy using a probit baseline and path-mass normalization.
- Special node-type prior now composes on top of ordinary baseline (`ordinary_prior_mean + mu_k`) so `mu_k` directly indicates uplift/downshift versus ordinary behavior.
- `EmFitMeta` records `ordinary_prior_mean` for downstream interpretation.

Per-tree visualization artifact:

- `EmFitPerTree` with node display score currently using `mean / variance`

---

## Terminology note

- We keep existing Rust module/type names under `src/em/*` for now to avoid broad refactor churn.
- Methodologically, this plan is MAP, not strict EM.
