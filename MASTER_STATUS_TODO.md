# MASTER STATUS TODO

## Goal

Introduce a tree-level master state machine:

- `TreeMasterStatus::WorkingOnTrajectory`
- `TreeMasterStatus::DeterminingBranchingNode`

and make branching orchestration explicit and event-sourced.

## Done in this step

1. Added `TreeMasterStatus` enum in `session.rs` with:
   - `WorkingOnTrajectory`
   - `DeterminingBranchingNode`
2. Added `tree_master_status: TreeMasterStatus` field to `Tree`.
3. Initialized `tree_master_status` as `WorkingOnTrajectory` in `Tree::new`.
4. Refactored `Tree` initialization to be logically empty:
   - `root_node_id: None`
   - `current_node_id: None`
   - `nodes: []`
   - `next_node_id: 0`
5. Updated rollout bootstrap to emit and apply root `CreateNode` when no loaded events exist.

## Next steps

1. **Master-status transitions in rollout**
   - When a trajectory ends, append the finished node to `leaf_node_ids`.
   - If number of finished trajectories is `< 16`, switch to `DeterminingBranchingNode`.
   - After selecting branching parent node, switch back to `WorkingOnTrajectory`.

2. **Branching-node selection logic**
   - Implement selection over existing one-child nodes (selection policy TBD).
   - Emit structural events in order:
     1) `CreateNode { parent_id: Some(selected_node_id) }`
     2) `SetCurrentNode { node_id: new_child_id }`

3. **Event model extension (if needed)**
   - Consider adding explicit master-status events (e.g., `SetTreeMasterStatus`) for full replay fidelity.
   - Alternatively, document why derived-on-replay status is sufficient.

4. **Replay invariants**
   - Assert root creation occurs exactly once and first among structural events for fresh trees.
   - Assert `current_node_id.is_some()` whenever `AddAction` is applied.
   - Assert leaf bookkeeping consistency when trajectory end is detected.

5. **Browser / tooling updates**
   - Update browsing/debug flows to display `tree_master_status` and branching metadata.
