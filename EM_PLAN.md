## Goal

Estimate per-node contribution values from leaf correctness outcomes, while jointly learning shared mode-level priors (`\mu_special_mode`) across many trees.

Contribution semantics: each node contributes a signed scalar to trajectory outcome; the trajectory latent outcome is the sum of node contributions along the path. Positive pushes toward success, negative pushes toward failure.

---

## Current modeling idea (updated)

Each judged leaf trajectory has a latent scalar score formed by summing node contributions along its path.
Observed outcome is binary success/failure derived from deterministic thresholding of that latent score.

Node priors are mode-dependent:

- Normal mode (`VerifierAndModeSummary::VerifierOff`): prior mean `0`
- Special modes (`VerifierOn`, `VerifierOnAndOverwriteLastStep`, `VerifierOnAndChangePlan`): prior mean `\mu_k`
- Shared mode means `\mu_k` are learned jointly across all trees with a zero-centered prior.

Each node also has a learned uncertainty scale parameter `log_std_i` that is regularized by a prior and enters the normalized trajectory sign constraint through path uncertainty.

---

## Chosen formulation so far

### A) Deterministic threshold constraints (chosen)

For each judged leaf `l`:

- `y_l in {+1, -1}` where `+1` is correct and `-1` is incorrect
- `s_l = sum_i x_{l,i} * m_i` where `m_i` is node contribution mean and `x_{l,i} in {0,1}` indicates whether node `i` is on leaf `l` path

Deterministic sign target:

- `y_l * s_l >= 0` (or margin variant `>= margin` once margin is decided)

### B) Slack variables for infeasibility (chosen)

To avoid impossible joint constraint sets, introduce per-leaf slack `xi_l >= 0`:

- `y_l * s_l >= -xi_l` (or `>= margin - xi_l` if using margin)

Optimization minimizes regularization terms plus aggregate slack penalty.

### C) Multi-tree fitting scope (chosen)

Fit is performed jointly on many trees at once so that shared `\mu_k` can be learned from pooled evidence.

### D) Output target (chosen)

Return per-node parameters:

- `mean` (signed contribution mean)
- `log_std` (learned, prior-regularized scale parameter)

---

## Remaining unresolved decisions (must finalize before implementation)

### 1) Prior hierarchy and variance hyperparameters

Model sketch:

- `mean_i ~ N(mu_{mode(i)}, sigma_mean^2)`
- `mu_k ~ N(0, sigma_mode^2)`
- `log_std_i ~ N(mu_log_std_mode(i), sigma_log_std^2)` (or zero-centered if mode-specific prior for log-std is not used)

Need concrete values or tuning policy for:

- `sigma_mean`
- `sigma_mode`
- `sigma_log_std`
- optional `mu_log_std_mode`

Status: fixed constants selected for now; revisit after first global fit diagnostics.

---

### 2) Validation/tuning policy for a single global fit

You selected single global fit. Remaining question: how to tune `lambda_slack` and sanity-check stability.

Current decision: **defer validation split/tuning to future work** due to implementation complexity.

Current practice:

- Perform single global fit over available trees.
- Manually inspect fitted distributions (`mean`, `log_std`, shared priors) and diagnostics.
- Adjust hyperparameters manually if distributions look unreasonable.

Future TODO (optional):

- Add held-out tree validation where shared/global parameters are frozen and validation-tree local node posteriors are fitted under those frozen globals.

---

### 3) Identifiability constraints

To prevent drift/confounding between node effects and mode means:

- **Option A:** rely on Gaussian priors only
- **Option B:** priors + explicit centering constraint on node residuals per mode
- **Option C:** priors + fixed intercept convention (e.g., root as intercept only)

Status: Option A selected (Gaussian priors only).

---

### 4) Optimization method

Candidate approaches:

- **Option A:** constrained optimization with slack + priors (deterministic)
- **Option B:** unconstrained penalty reformulation (replace hard constraints with hinge-like loss + priors)
- **Option C:** augmented Lagrangian / ADMM variant

Status: Option A selected (constrained optimization with slack + priors).

---

### 5) Convergence and stopping criteria

Status: max-iterations-only selected.

Confirmed fixed default: `max_iterations = 100`.

---

### 6) Numerical stability rules

Need explicit rules for:

- minimum/maximum clamps for `log_std`
- handling near-zero or exploding prior scales
- handling effectively separable labels where slack collapses to zero for many points

Recommended defaults:

- clamp `u_i = log_std_i` to `[-4.0, 2.0]` (std range about `[0.018, 7.39]`)
- assert all sigma hyperparameters are finite and `> 0`
- assert `eps >= 1e-8`

Confirmed: use the above clamp range and safeguards.

---

### 7) Fail-fast assertions (implementation contract)

Given project coding style, decide required asserts:

- at least one judged leaf exists
- each judged leaf path resolves to root without broken parent chain
- no duplicated node ids in a single path
- all referenced node ids are in bounds and index-consistent
- mode tag exists when required by fitting rule

Status: pending final confirmation with added leaf/judgment set-equality asserts below.

---

### 8) API boundary: Tree method vs global trainer

Need to decide the exact responsibility split.

- **Tree-level method (recommended):**
  - extract per-tree sufficient statistics / sparse path encoding
- **Global estimator:**
  - consume many trees and update shared `mu_k` + per-node (`mean`, `log_std`)

Recommended: keep estimator outside `Tree`; `Tree` only provides extraction helpers.

Confirmed boundary:

- Keep estimator/fitter outside `Tree`.
- Use a separate struct to interact with trees and store fitted distributions.

---

### 9) Serialization and downstream usage

What artifacts should be persisted?

- per-node signed contribution mean
- per-node `log_std`
- shared `mu_k`
- slack diagnostics (`sum_xi`, `num_positive_xi`, largest violators)
- fit diagnostics (objective trace, convergence flag)

Recommended schema (minimum):

- per-node: `{global_node_id, tree_question_id, node_id, mean, log_std, mode}`
- global: `{mu_k_by_mode, nu_k_by_mode, lambda_slack, sigma_mean, sigma_mode, sigma_log_std, eps, max_iterations}`
- diagnostics: `{objective_trace, converged_flag=false|true, final_train_sign_accuracy, final_val_sign_accuracy, mean_slack_train, mean_slack_val}`

Confirmed persisted diagnostics and schema direction.

---

## Proposed immediate next step

Decide items in this order (to unblock implementation fastest):

1. Margin + slack penalty (#1, #2)
2. Path + leaf inclusion rules (#3, #4)
3. Prior/variance + identifiability (#5, #6)
4. Optimizer + convergence (#8, #9, #10)
5. API boundary + persistence (#12, #13)

---

## Decision log (confirmed)

1. Constraint margin convention: **zero margin**
   - Use `y_l * s_l >= -xi_l`

2. Slack penalty form: **L2**
   - Use `lambda_slack * sum_l xi_l^2`
   - Recommended initial value: `lambda_slack = 1.0`

3. Path inclusion rule:
   - Include root node: **yes**
   - Include terminal leaf node: **yes**
   - Include nodes with `step.verifier_and_mode_summary == None`: **yes**
   - Fitting-time mode mapping for `None`: **treat as VerifierOff**

4. Leaf eligibility rule:
   - Prefer `leaf_node_judgments` (labels source)
   - Require judgment ids and leaf ids to match before fitting (assertion)

5. Prior hierarchy and variances: **fixed values for now**
   - Recommended initialization:
     - `sigma_mean = 1.0`
     - `sigma_mode = 1.0`
     - `sigma_log_std = 1.0`
     - `mu_log_std_mode = 0.0` (implies prior center at `std = exp(0)=1`)

6. Identifiability: **Gaussian priors only**

7. Joint fitting scope: **single global fit over all selected trees**

8. Optimization method: **constrained optimization with slack + priors**

9. Convergence / stopping: **max iterations only**
   - Recommended initial value: `max_iterations = 100`

---

## Notes on leaf/judgment consistency

Current rollout logic registers and judges the current leaf when a trajectory ends, using the same `leaf_node_id` in sequence.
However, abrupt interruption between those two events can leave temporary mismatch in persisted logs.

For fitting, enforce fail-fast consistency:

- `assert_eq!(tree.leaf_node_ids.len(), tree.leaf_node_judgments.len())`
- `assert!(leaf_node_ids set == leaf_node_judgments.keys() set)`

If these assertions fail, treat logs as incomplete and repair/replay before fitting.

---

## Objective function block (implementation draft)

This block is the code-facing mathematical specification for the chosen deterministic + slack formulation.

### Index sets and data

- Trees: `t in {1..T}`
- Nodes in tree `t`: `i in N_t`
- Judged leaves in tree `t`: `l in L_t`
- Global node index set: `N = union_t N_t`
- Global judged leaf set: `L = union_t L_t`

Observed labels:

- `y_l in {+1, -1}` (`+1` for correct, `-1` for incorrect)

Path indicator:

- `x_{l,i} in {0,1}`
- `x_{l,i} = 1` iff node `i` is on judged leaf `l` path
- By decision: path includes root and terminal leaf nodes.

Mode mapping:

- `mode(i) in {VerifierOff, VerifierOn, VerifierOnAndOverwriteLastStep, VerifierOnAndChangePlan}`
- If `step.verifier_and_mode_summary == None`, map to `VerifierOff`.

### Parameters to optimize

- Per-node contribution mean: `m_i` for each `i in N`
- Per-node log std: `u_i` for each `i in N` (where `std_i = exp(u_i)`)
- Shared mode means for contributions: `mu_k` for each mode `k`
- Shared mode means for log-std priors: `nu_k` for each mode `k`
- Per-leaf slack: `xi_l >= 0` for each `l in L`

### Deterministic constraints with uncertainty coupling (chosen)

Leaf score mean:

- `mu_l = sum_{i in N} x_{l,i} * m_i`

Leaf score variance under independent node contributions:

- `var_l = sum_{i in N} x_{l,i} * exp(2 * u_i)`
- `sigma_l = sqrt(var_l)`

Normalized zero-margin sign constraints with slack:

- `y_l * mu_l / (sigma_l + eps) >= -xi_l`
- `xi_l >= 0`

Equivalent form:

- `xi_l >= - y_l * mu_l / (sigma_l + eps)`
- `xi_l >= 0`

Interpretation:

- Larger uncertainty (`sigma_l`) weakens effective margin and increases required slack.
- This couples `log_std` into fitting while keeping deterministic sign-constraint semantics.

### Prior regularization terms

Contribution-mean prior:

- `m_i ~ N(mu_{mode(i)}, sigma_mean^2)`

Mode-mean prior:

- `mu_k ~ N(0, sigma_mode^2)`

Log-std prior:

- `u_i ~ N(nu_{mode(i)}, sigma_log_std^2)`

Mode prior for log-std centers:

- `nu_k ~ N(mu_log_std_mode, sigma_mode^2)`
- Current default center: `mu_log_std_mode = 0.0`

### Optimization objective (minimization)

`J = J_constraints + J_mean_prior + J_mode_prior + J_log_std_prior + J_log_std_mode_prior`

where:

- `J_constraints = lambda_slack * sum_{l in L} xi_l^2`
- `J_mean_prior = (1 / (2 * sigma_mean^2)) * sum_{i in N} (m_i - mu_{mode(i)})^2`
- `J_mode_prior = (1 / (2 * sigma_mode^2)) * sum_k mu_k^2`
- `J_log_std_prior = (1 / (2 * sigma_log_std^2)) * sum_{i in N} (u_i - nu_{mode(i)})^2`
- `J_log_std_mode_prior = (1 / (2 * sigma_mode^2)) * sum_k (nu_k - mu_log_std_mode)^2`

Subject to:

- `y_l * (sum_i x_{l,i} m_i) / (sqrt(sum_i x_{l,i} exp(2 * u_i)) + eps) + xi_l >= 0` for all `l`
- `xi_l >= 0` for all `l`

### Fixed initial hyperparameters (current default)

- `sigma_mean = 1.0`
- `sigma_mode = 1.0`
- `sigma_log_std = 1.0`
- `mu_log_std_mode = 0.0`
- `eps = 1e-6`
- `lambda_slack = 1.0` (recommended start)
- `log_std clamp: u_i in [-4.0, 2.0]`

`lambda_slack` is still to be set explicitly (recommended to tune on a small validation slice first).

### Notes for implementation

- This formulation now couples `u_i` into constraints via path-level denominator `sigma_l`.
- Keep `eps > 0` to avoid divide-by-zero in degenerate paths.
- For numerical stability, clamp `u_i` into a bounded interval during optimization (exact clamp range remains to be finalized in stability decisions).
