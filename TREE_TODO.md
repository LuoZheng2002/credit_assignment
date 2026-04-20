# TREE TODO

## Completed

1. Emit `SetCurrentNode` events from actual transition logic
   - Implemented in rollout after `PlannerUpdatePlan`: emit `CreateNode` then `SetCurrentNode`.
   - Cursor movement is now event-sourced instead of implicit.

3. Make replay fully event-complete in pipeline
   - Pipeline loads per-question `Vec<TreeUpdateEvent>` and replays via `tree.apply_event(event)`.
   - Structural/cursor/action events all use the same replay path.

5. Add event-order invariants (assert-first style)
   - `CreateNode` checks parent existence, node-id uniqueness, and node-id ordering.
   - `SetCurrentNode` checks target existence.
   - `AddAction` appends on current cursor only.

## Unfinished next steps

2. Emit and apply `CreateNode` from rollout branching logic
   - Add real branching decisions in rollout (currently still effectively single-path runtime).
   - On branch creation, emit `CreateNode { question_id, node_id, parent_id }` and apply it through `Tree::apply_event`.
   - Then emit `SetCurrentNode` for the chosen working node.

4. Remove remaining implicit linear assumptions
   - Audit rollout/session code for any assumptions that current node is always root path only.
   - Replace with explicit tree cursor + events.
   - Recent cleanup already done: removed obsolete `Tree::last_action()` helper that had parent-only fallback behavior.

6. Optional follow-up: migrate `TrajectoryState` construction entrypoint
   - Add `Tree::build_trajectory_state()` helper to standardize `TrajectoryState::from_tree(tree)` call pattern.
   - This reduces repeated boilerplate and clarifies ownership of state reconstruction.
