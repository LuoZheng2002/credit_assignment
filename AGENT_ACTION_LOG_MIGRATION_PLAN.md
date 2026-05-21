# AGENT_ACTION_LOG_MIGRATION_PLAN

## Goal

Make `TreeAction[]` the canonical source of truth for agent rollout state and final results. Treat `CompletedTree` as a derived view reconstructed from action logs (instead of a separately persisted primary artifact).

This is intentionally a breaking change.

## Why This Change

- Remove dual-write consistency risk between action log tables and `CompletedTree` store.
- Eliminate delete/drop semantics for per-question logs after completion.
- Simplify recovery semantics: one authoritative event stream per question.
- Align with the direct-tool pattern (load full log, append in memory, upsert full array).

## Current State (As-Is)

- `src/agent/rollout_batch.rs` currently:
  - Writes incremental `TreeAction` events to `SqliteSessionLogStore` (`SqliteTableArrayStore`).
  - Writes final `CompletedTree` to `CompletedTreeStore`.
  - Deletes per-question action table once a tree is completed.
- Downstream stages (`EM`, advantage composition, formatted/tokenized training set, session browser) consume `CompletedTreeStore` via `AssetFileTrees`.

## Target State (To-Be)

- Persistent authoritative artifact: `SqliteStore<usize, TreeActionLog>`.
- New schema mirrors direct-tool style:
  - `TreeActionLog { question: SingleDatasetQuestion, actions: Vec<TreeAction> }`
  - primary key is `SingleDatasetQuestion.id`.
- One DB entry per question id; payload is full log snapshot.
- Rollout syncing writes full snapshots (upsert entire `TreeActionLog`).
- Completion is represented by terminal event (`TreeComplete`) in the array; no delete needed.
- `CompletedTree` is reconstructed on demand (or materialized as cache) from `TreeActionLog`.

---

## Feasibility Analysis

### Can `CompletedTree` be reconstructed from `TreeAction[]`?

Short answer: **Yes, with explicit assumptions**.

### What is recoverable exactly

From action array + deterministic reducer (`Tree::apply_action`):

- `trajectory: Tree` (all nodes, edges, node logs, leaf judgments, correctness ratio, completion flag)
- `id` (from `question_id` in actions / store key)
- `step_quality_ratio` (`tree.get_step_quality_ratio()`)
- `failed_and_aborted_ratio` (`tree.get_failed_and_aborted_ratio()`)

### Missing fields resolution (adopted)

Adopt the direct-tool pattern by storing question payload inside the canonical record:

- `TreeActionLog.question: SingleDatasetQuestion` carries:
  - `id`
  - `question`
  - `final_answer` (reference answer)

With this schema, `CompletedTree.question` and `CompletedTree.correct_answer` are recoverable without external dataset lookup.

### Required assumptions for full equivalence

- `TreeActionLog.question.id` must equal sqlite key and all action `question_id` values.
- `TreeActionLog.question` must be immutable after first creation.
- `actions` order must be append-only event order.

### Determinism and correctness notes

- Reducer determinism is strong: `Tree::apply_action` is pure w.r.t. action sequence order.
- `JudgeLeafCorrectness` embeds `is_correct` and `correct_answer`, so no external judge rerun is needed.
- If action arrays are truncated/corrupted, reconstruction will panic/fail fast due to existing assertions. This is acceptable if treated as data-integrity failure.

### Feasibility verdict

**High feasibility**, with moderate migration scope.

- Core rollout logging change: straightforward.
- Main migration cost: downstream code currently assumes persisted `CompletedTreeStore`.
- Critical design decision: where to source `question` + `correct_answer` during reconstruction.

---

## Migration Design

## 1) Canonical Store Model

Introduce canonical structs/type aliases:

- `pub struct TreeActionLog { question: SingleDatasetQuestion, actions: Vec<TreeAction> }`
- `type ActionLogStore = SqliteStore<usize, TreeActionLog>`

and asset wrapper (new module, e.g. `agent/action_log_schema.rs`):

- path helpers and version tracking
- `fetch()` returns store handle
- optional `load_all_completed()` helper (filters logs with terminal `TreeComplete`)

## 2) Reconstruction API

Add explicit reconstruction entrypoints:

- `reconstruct_tree(log: &TreeActionLog) -> Tree`
- `reconstruct_completed_tree(log: &TreeActionLog) -> CompletedTree`

Behavior:

- Start with `Tree::new(log.question.id, log.question.question.clone(), log.question.final_answer.clone())`.
- Apply actions in order.
- Validate completion (`tree.completed == true`) when caller requests completed-only view.
- Compute derived ratio fields from reconstructed tree.

## 3) Rollout Write Semantics

In `rollout_batch` receiver path:

- Maintain in-memory `IndexMap<usize, TreeActionLog>`.
- On each `LogOrTree::Action`, append in memory and upsert full `TreeActionLog`.
- Stop writing `CompletedTree` as primary artifact.
- Completion event (`TreeComplete`) marks done; no delete.

Optional safety:

- Keep in-memory `completed_ids` set to avoid post-completion writes.

## 4) Read Model for Downstream Pipeline

Replace direct `AssetFileTrees.fetch()` reads with one of:

- On-demand iterator that reconstructs `CompletedTree` from canonical `TreeActionLog`.
- Or one-way materialized cache file generated from logs (non-canonical, reproducible).

Recommendation:

- Keep a temporary compatibility layer exposing `CompletedTree` iterator while internals source from action logs.

## 5) Compatibility Strategy (Breaking Change Management)

Because this is breaking and formats differ, do not reuse old file schema silently.

- Use new file path suffix (e.g. `_action_log_v2.sqlite`).
- Add schema/version marker in tracking JSON.
- Optional one-time migration command from old artifacts:
  - If `CompletedTree` exists and logs are missing, migration cannot reconstruct intermediate actions.
  - Therefore migration target should prioritize existing session logs where present.

## 6) Data Integrity Rules

Define invariants and validate:

- `TreeActionLog.question.id` equals store key.
- All actions in an entry share the same `question_id` and match `TreeActionLog.question.id`.
- `TreeActionLog.question` content (`question`, `final_answer`) never changes after first write.
- `TreeComplete` appears at most once and only at the end.
- No actions after completion.
- Reconstruction must not panic for canonical completed entries.

## 7) Rollout Loop API/Logic Alignment with Direct Tool

Change `src/agent/rollout_loop.rs::rollout` to mirror `src/direct_tool/direct_rollout.rs::rollout` style:

- Input argument pattern:
  - `question: SingleDatasetQuestion`
  - `rollout_store: SqliteStore<usize, TreeActionLog>`
  - other existing runtime dependencies (`llm_callable`, `client`, `rng`)
- State loading pattern:
  - `let mut action_log = rollout_store.get(question.id).await?...unwrap_or(TreeActionLog { question: question.clone(), actions: vec![] })`
  - reconstruct tree from `action_log.actions`
- Loop/write pattern:
  - produce `new_actions`
  - append each action to `action_log.actions`
  - `rollout_store.upsert(question.id, &action_log).await?` (full snapshot sync)
- Completion detection by terminal event / reconstructed tree completion, not by external `CompletedTree` write/delete.

Violations should fail loudly and identify the offending question id.

---

## Impacted Areas

High-impact modules to refactor:

- `src/agent/rollout_batch.rs`
- `src/agent/rollout_loop.rs` (signature and logic will mirror direct-tool rollout style)
- `src/agent/sqlite_rollout_log.rs` (replace table-array store alias)
- `src/agent/tree_schema.rs` and `AssetFileTrees` behavior
- `src/em/em_schema.rs` (input source currently `CompletedTreeStore`)
- `src/agent/advantage_composition.rs`
- `src/training_set/training_set_formatted.rs`
- `src/bin/bin_browse_session.rs`
- `src/training_set/training_set_generation.rs`

---

## Proposed Phased Execution

## Phase 0: Guardrails and Spec

- Write and agree on canonical invariants for `TreeActionLog` records.
- Define reconstruction contract and failure modes.
- Lock the chosen metadata strategy: embed immutable `SingleDatasetQuestion` in `TreeActionLog`.

Exit criteria:

- Spec approved for canonical event schema and reconstruction behavior.

## Phase 1: Canonical Store + Dual Read

- Implement `TreeActionLog` schema and new action-log asset/store.
- Implement reconstruction helpers.
- Add adapter that exposes reconstructed `CompletedTree` iterator to existing consumers.

Exit criteria:

- Existing downstream modules can read reconstructed trees without behavior drift.

## Phase 2: Rollout Writer Migration

- Change rollout writer to full-snapshot upserts of `TreeActionLog`.
- Remove per-question table append/drop logic.
- Completion represented only by events.
- Refactor `src/agent/rollout_loop.rs` function signature/logic to mirror `src/direct_tool/direct_rollout.rs`.

Exit criteria:

- Interrupt/resume works from canonical logs only.
- No delete/drop operations needed for normal flow.

## Phase 3: Downstream Consumer Cutover

- Move `EM`, advantage, training-set generation, and browser to canonical-log-backed reads.
- Keep optional materialized tree cache as derived artifact if needed for performance.

Exit criteria:

- Full pipeline runs without requiring `CompletedTree` as primary persisted source.

## Phase 4: Cleanup and Breaking Removal

- Remove legacy `SqliteSessionLogStore` path and old assumptions.
- Remove obsolete migration bridges if not needed.
- Document new storage model in README/tool docs.

Exit criteria:

- Codebase has a single source of truth: action log arrays.

---

## Risk Assessment

- **Performance risk (medium):** full-array upsert per action can be write-heavy for long trajectories.
  - Mitigation: batch writes (e.g. per action batch or timed flush).
- **Compatibility risk (high):** existing tree assets and downstream tools expect `CompletedTreeStore`.
  - Mitigation: reconstruction adapter and phased cutover.
- **Data duplication risk (medium):** storing `SingleDatasetQuestion` per log entry increases payload size.
  - Mitigation: acceptable tradeoff for self-contained recoverability; optionally compress at storage layer later.
- **Integrity risk (low-medium):** malformed event streams can fail reconstruction.
  - Mitigation: strict invariants + validation tooling.

---

## Validation Plan

Core checks:

- For a sample run, compare legacy `CompletedTree` vs reconstructed `CompletedTree` per question:
  - node count, leaf ids/judgments, correctness ratio, completion flag
  - `step_quality_ratio`, `failed_and_aborted_ratio`
- Resume test:
  - kill rollout mid-run, restart, verify continued progression without loss/duplication.
- End-to-end pipeline test:
  - EM fit, advantage composition, training set generation, and browser all operate from canonical logs.

Optional robust checks:

- Add checksum over action arrays per question.
- Add repair script to detect and quarantine invalid entries.

---

## Recommendation

Proceed with migration. The model is viable and simplifies persistence semantics. With `TreeActionLog { question: SingleDatasetQuestion, actions }`, full equivalence to current `CompletedTree` is achievable without external dataset dependency.

Recommended default:

- Keep canonical `TreeActionLog` as source of truth.
- Use sqlite key = `SingleDatasetQuestion.id` and enforce id consistency against action `question_id`.
- Refactor `src/agent/rollout_loop.rs` to match direct-tool rollout argument shape and full-snapshot upsert logic.

---

## Decision Log (Fill Before Implementation)

Update this section by replacing `TBD` with your selected option label.

### D1) Rollout function error contract

- **A:** Return `Result<(), String>` from `rollout_loop::rollout` and propagate errors.
- **B:** Keep `()` and fail via panics/asserts on invariant violations.
- **C:** Return `Result` for storage/IO errors, keep asserts for invariant violations.
- **Chosen:** B

### D2) Action-log sync frequency

- **A:** Upsert full `TreeActionLog` after every emitted action.
- **B:** Upsert once per `produce_actions_from_state` batch.
- **C:** Upsert on timed interval and on completion.
- **Chosen:** B

### D3) Canonical completion criterion

- **A:** Completed iff last action is `TreeAction::TreeComplete`.
- **B:** Completed iff reconstruction yields `tree.completed == true`.
- **C:** Require both A and B (strictest).
- **Chosen:** B

### D4) Incomplete log handling in downstream readers

- **A:** Skip incomplete logs, warn.
- **B:** Hard-fail pipeline on first incomplete log.
- **C:** Include incomplete logs in diagnostics only; exclude from training/EM.
- **Chosen:** C (and produce warnings for incomplete logs excluded from training/EM)

### D5) Legacy compatibility window

- **A:** Keep `AssetFileTrees` adapter for one release cycle, then remove.
- **B:** Keep indefinitely as non-canonical compatibility facade.
- **C:** Remove immediately in same breaking change.
- **Chosen:** C

### D6) Migration/backfill policy

- **A:** No backfill; new runs only on `AssetFileActionLogs`.
- **B:** Add migration from existing session logs when present; ignore `CompletedTree`-only artifacts.
- **C:** Add migration tool with best-effort from both logs and `CompletedTree`.
- **Chosen:** A

### D7) Id consistency enforcement

- **A:** Hard-fail on any mismatch (`store key`, `question.id`, action `question_id`).
- **B:** Quarantine bad entry, continue processing others.
- **C:** Auto-rewrite mismatches to store key when safe, otherwise quarantine.
- **Chosen:** A (by the way, see if action's question_id can be safely removed given we do not aggregate actions through channels anymore)

### D8) Batch writer concurrency model

- **A:** Single writer task per question (serialized per key).
- **B:** Central writer task with in-memory `IndexMap<id, TreeActionLog>`.
- **C:** Hybrid: per-question buffer + central flush.
- **Chosen:** A (now the rollout function should interact directly with the SqliteStore and update the database without delegating to the rollout_batch function)

### D9) SQLite path/versioning

- **A:** New filename suffix (e.g. `_action_log_v2.sqlite`) and leave legacy untouched.
- **B:** Reuse existing filename with schema marker check and hard fail on mismatch.
- **C:** Reuse filename with in-place migration.
- **Chosen:** B

### D10) Derived `CompletedTree` materialization strategy

- **A:** Pure on-demand reconstruction only (no persisted derived cache).
- **B:** On-demand reconstruction plus optional materialized cache command.
- **C:** Always materialize derived `CompletedTree` cache after rollout.
- **Chosen:** A (like src/direct_tool/direct_rollout.rs, the tree should be reconstructed each time the action log changes without maintaining a state and being mutable, but if the action log does not change, we may cache the tree wherever convenient)

When all `Chosen` fields are filled, implementation can proceed immediately.

---

## Appendix: File-by-File Signature and Type Change Checklist

This appendix lists concrete code-level changes to execute the migration, without prescribing implementation details beyond types/signatures and call flow.

## A) New/Updated Canonical Log Schema

### `src/agent/sqlite_rollout_log.rs`

- Replace legacy alias:
  - from `SqliteSessionLogStore = SqliteTableArrayStore<usize, TreeAction>`
  - to canonical aliases similar to:
    - `pub type TreeActionLogStore = SqliteStore<usize, TreeActionLog>;`
- Keep/rename path helper:
  - `get_rollout_log_path(...)` may move to new filename suffix (recommended versioned path).

### `src/agent/tree_action_log.rs` (new file recommended)

- Add struct mirroring direct-tool pattern:

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TreeActionLog {
    pub question: SingleDatasetQuestion,
    pub actions: Vec<TreeAction>,
}
```

- Imports:
  - `crate::agent::single_dataset::SingleDatasetQuestion`
  - `crate::agent::tree_action::TreeAction`

### `src/agent/mod.rs`

- Export new module:
  - `pub mod tree_action_log;`

## B) Rollout Loop API Alignment with Direct Tool

### `src/agent/rollout_loop.rs`

- Change function signature from event-stream style to store-driven style (mirror `direct_rollout`):
  - **Current style:** accepts `question_id`, `question`, `reference_answer`, preloaded `Vec<TreeAction>`, and emits over channel.
  - **Target style:**
    - `question: SingleDatasetQuestion`
    - `rollout_store: TreeActionLogStore` (or `SqliteStore<usize, TreeActionLog>`)
    - runtime deps (`llm_callable`, `client`, `rng`)
    - return type can be `()` (direct-tool style) or `Result<(), String>`.

- Internal flow checklist:
  - `get(question.id)` -> existing `TreeActionLog` or initialize `{ question: question.clone(), actions: vec![] }`
  - reconstruct `Tree` by applying `action_log.actions`
  - while not completed:
    - call `produce_actions_from_state(...)`
    - append each produced action to in-memory `action_log.actions`
    - apply action to tree
    - `upsert(question.id, &action_log)` (full snapshot)

- Invariants to enforce in function:
  - existing log question id equals `question.id`
  - action `question_id` values equal `question.id`

## C) Batch Orchestration Refactor

### `src/agent/rollout_batch.rs`

- Remove `LogOrTree` dual channel model once loop becomes store-driven.
- Replace `SqliteSessionLogStore` usage with `TreeActionLogStore`.
- Submission task should pass:
  - full `SingleDatasetQuestion`
  - cloned store handle
  - runtime deps
- Progress tracking:
  - derive completion by reading/reconstructing logs and checking terminal condition (`TreeComplete` or `tree.completed`).
- Remove delete/drop behavior entirely.
- Keep Ctrl+C handling and unfinished-question scheduling behavior.

## D) Reconstruction Helpers for Derived CompletedTree

### Recommended new module: `src/agent/tree_reconstruction.rs`

- Add helper signatures:
  - `pub fn reconstruct_tree(log: &TreeActionLog) -> Tree`
  - `pub fn reconstruct_completed_tree(log: &TreeActionLog) -> CompletedTree`
  - optional `pub fn is_completed(log: &TreeActionLog) -> bool`

- `reconstruct_completed_tree` fields:
  - `id = log.question.id`
  - `question = log.question.question.clone()`
  - `correct_answer = log.question.final_answer.clone()`
  - `trajectory = reconstructed tree`
  - ratio fields computed from reconstructed tree methods

## E) Tree Asset Interface Transition

### `src/agent/tree_schema.rs`

- Keep `CompletedTree` struct (downstream compatibility), but shift persistence role:
  - no longer primary write target from rollout.
- **Decision made:** choose option 2.
  - Introduce `AssetFileActionLogs` as canonical asset.
  - Downstream systems read action logs via this asset and reconstruct `CompletedTree` through adapter/helpers.
  - `AssetFileTrees` may remain as a temporary compatibility facade, but should not be canonical.

## F) Downstream Consumer Adaptation

### `src/em/em_schema.rs`

- Replace assumptions of directly persisted `CompletedTreeStore` input.
- Input should come from reconstructed completed trees iterator/vector.
- Keep public EM output schema unchanged.

### `src/agent/advantage_composition.rs`

- `compose_advantage(...)` input can remain `CompletedTree`-based, but source must be reconstructed trees.
- Ensure tree id set logic still holds.

### `src/training_set/training_set_formatted.rs`

- Replace `AssetFileTrees.fetch()` dependency path with reconstructed trees provider.
- No change needed to formatted sample schema.

### `src/training_set/training_set_generation.rs`

- Function signatures can stay `CompletedTree`-based if reconstruction happens upstream.

### `src/bin/bin_browse_session.rs`

- Replace tree loading path to read reconstructed `CompletedTree` (or on-demand reconstruction per selected question).

## G) CLI / Entry Point Wiring

### `src/bin/bin_tree.rs`

- Ensure call chain uses refactored rollout APIs.
- No expected CLI arg changes.

## H) Versioning and Pathing

- Introduce versioned action-log sqlite path (recommended):
  - e.g. `..._rollout_action_log_v2.sqlite`
- Add/update tracking JSON schema to include:
  - action log schema version
  - optional dataset hash for auditability

## I) Verification Checklist During Implementation

- Compile-time:
  - no remaining references to `SqliteTableArrayStore` for agent rollout logs
  - no remaining append/drop-table logic for agent rollout logs
- Runtime:
  - interrupt and resume retains progress accurately
  - reconstructed `CompletedTree` matches legacy outputs on fixture subset
- Data invariants:
  - key/id consistency (`store key == question.id == action.question_id`)
  - no post-completion actions
