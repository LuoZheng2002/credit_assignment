# TREE TODO

## Unfinished next steps

1. Emit `SetCurrentNode` events from actual transition logic
   - When rollout decides to move the cursor (e.g., after step finalization / branch switch), emit `TreeUpdateEvent::SetCurrentNode` explicitly.
   - Keep cursor movement event-sourced instead of implicit in in-memory logic.

2. Emit and apply `CreateNode` from rollout branching logic
   - Add real branching decisions in rollout (currently still effectively single-path runtime).
   - On branch creation, emit `CreateNode { question_id, node_id, parent_id }` and apply it through `Tree::apply_event`.
   - Then emit `SetCurrentNode` for the chosen working node.

3. Make replay fully event-complete in pipeline
   - Ensure loaded per-question event streams include all structural/cursor events (`CreateNode`, `SetCurrentNode`, `AddAction`).
   - Keep replay path as `tree.apply_event(event)` only.

4. Remove remaining implicit linear assumptions
   - Audit rollout/session code for any assumptions that current node is always root path only.
   - Replace with explicit tree cursor + events.

5. Add event-order invariants (assert-first style)
   - Assert valid ordering during apply, for example:
     - `CreateNode` parent must already exist.
     - `SetCurrentNode` target must already exist.
     - `AddAction` target node must be current cursor (already implied by current event shape).

6. Optional follow-up: migrate `TrajectoryState` construction entrypoint
   - Add `Tree::build_trajectory_state()` helper to standardize `TrajectoryState::from_tree_node(question.clone(), tree.current_node.clone())` call pattern.
   - This reduces repeated boilerplate and clarifies ownership of state reconstruction.
