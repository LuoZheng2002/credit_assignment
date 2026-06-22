# EXTSORT Plan for `ActionLogStore`

## Goal
Introduce an `extsort`-backed action log backend alongside the existing `redb` backend, with the long-term goal of making action-log writes cheap during rollout and reads fast and sequential during tree reconstruction.

For now, the backend choice should remain **hardcoded to `redb`** in code until the `extsort` backend is fully implemented and validated.

---

## Current state

`src/tree_action_log.rs` currently exposes a single public store API:

- `ActionLogStore<M, S>`
- `get_keys()`
- `load_table_sorted()`
- `load_or_init_table_sorted()`
- `append_at()`
- `commit_pending()` via `ActionStoreAdapter`

The current implementation is effectively `redb`-only:

- rollout writes actions through `ActionStoreAdapter`
- readers in `src/training_set.rs`, `src/browse_trees/mod.rs`, `src/bin/bin_browse_trees.rs`, and related code load actions via `load_table_sorted()`
- the order assumption is per-question `action_index` ordering after load

The ext-sort backend does **not** exist yet.

---

## Desired backend behavior

Each action row should be identified by:

1. `question_flat_id`
2. `action_index` within the action array for that question

Sort precedence:

- `question_flat_id` first
- `action_index` second

Behavioral expectations:

- rollout can append rows in arbitrary order
- a sort/finalization step orders rows before reconstruction
- reconstruction code reads data sequentially in sorted order
- duplicate keys with conflicting payloads must still be treated as an error

---

## Hardcoded backend choice

Until the new backend is complete, the code should remain hardcoded to `redb`.

Recommended policy:

- keep `ActionLogStore` construction pointing to `RedbActionLogStore`
- do not add runtime configuration or CLI toggles yet
- do not branch call sites on backend type
- add `extsort` as an implementation detail later, behind the same public API

This avoids half-finished backend selection logic while the new storage path is being built.

---

## Implementation plan

### Phase 1: isolate the store interface

Keep the current public API stable and introduce a backend abstraction internally.

Likely shape:

- `ActionLogStore<M, S>` as the public wrapper
- internal backend enum or trait
- `RedbActionLogStore<M, S>` as the active backend
- `ExtSortActionLogStore<M, S>` added later

This should preserve existing call sites while making backend swaps possible later.

---

### Phase 2: define the ext-sort record format

Store each row as:

- `question_flat_id`
- `action_index`
- serialized `DirectTreeAction<M>`

Use a sortable prefix that makes ordering deterministic and cheap.

Suggested encoding:

- `u64` big-endian `question_flat_id`
- `u64` big-endian `action_index`
- serialized payload bytes

This gives the desired lexicographic order for external sorting.

---

### Phase 3: append-only write path for rollout

In `src/rollout.rs`, the ext-sort backend should support raw appends during rollout.

The write path should:

- append rows in the order they are produced
- avoid per-write sorting
- flush efficiently
- preserve all data needed for later sorting and reconstruction

The point is to make rollout writes cheap and sequential.

---

### Phase 4: add finalization / sort step

Before readers consume the store, run one external sort pass.

That pass should:

- sort by `(question_flat_id, action_index)`
- collapse or reject duplicate keys
- write out a sorted read-optimized representation
- make sequential reading fast afterward

Best place for this step is likely at the end of rollout, not on first read.

That keeps read paths simple and avoids hidden latency spikes.

---

### Phase 5: make readers consume the sorted output

The main consumers are:

- `src/training_set.rs`
- `src/browse_trees/mod.rs`
- `src/bin/bin_browse_trees.rs`
- other call sites that use `load_table_sorted()`

These readers should continue to ask for the actions for one `question_flat_id` and receive them in `action_index` order.

If the ext-sort backend is implemented correctly, the read side should not need to know how sorting happened.

---

### Phase 6: preserve existing semantics

Match current redb behavior where it matters:

- return all initialized question keys from `get_keys()`
- reject conflicting duplicate rows
- keep tree reconstruction behavior unchanged
- keep current public method names and signatures if possible

The backend change should be invisible to higher-level logic.

---

## Files likely to change later

### Storage layer
- `src/tree_action_log.rs`

### Rollout write path
- `src/rollout.rs`

### Readers / reconstruction
- `src/training_set.rs`
- `src/browse_trees/mod.rs`
- `src/bin/bin_browse_trees.rs`
- possibly `src/get_accuracy.rs`

### Path/config plumbing if backend-specific paths become necessary
- `src/jinja_directories.rs`
- `config/directories/action_logs_*`

At the moment, no backend toggle is planned; the code stays on `redb`.

---

## Tests to add

### Ordering tests
Verify that rows are sorted by:

1. `question_flat_id`
2. `action_index`

### Round-trip tests
Append actions in arbitrary order, finalize, then verify that reads return the expected action sequence.

### Duplicate detection tests
Ensure:

- same key + same payload is accepted
- same key + different payload is rejected

### Integration tests
Run the full rollout-to-reconstruction path using the existing `redb` backend to ensure the refactor does not regress behavior.

---

## Recommended migration strategy

1. Keep the current `redb` backend as the only active implementation.
2. Refactor the internal structure so a second backend can be added without changing call sites.
3. Implement ext-sort write/finalize/read paths.
4. Add tests and validate sequential read performance.
5. Only then consider making backend selection configurable.

---

## Non-goals for now

- runtime backend switching
- CLI flags for backend selection
- changing the tree reconstruction semantics
- changing the serialized `DirectTreeAction` schema unless required by the sorter

---

## Summary

The short-term plan is to keep the code path hardcoded to `redb`, while preparing `src/tree_action_log.rs` for an `extsort` backend that writes unsorted rows during rollout, sorts them by `(question_flat_id, action_index)` before reading, and preserves the current store API for all consumers.
