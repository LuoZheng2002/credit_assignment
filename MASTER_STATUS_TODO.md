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
6. Refactored trajectory production logic:
   - extracted `produce_working_trajectory(...)`
   - extracted `determine_branching_node(...)`
   - inlined master-status dispatch loop into `rollout(...)`
7. `build_new_operations(...)` now derives `question_id` from `session_state.source_tree.question_id` to avoid argument duplication.
8. Sender passing is now by reference in extracted trajectory functions.
9. Implemented branching-node selection with trajectory-weight aggregation and weighted sampling:
   - each finished trajectory contributes uniform weight over non-leaf nodes summing to 1
   - shared nodes aggregate weight contributions across trajectories
   - sampling uses `WeightedIndex`
10. Replaced single-child topology with dual branch slots per node:
   - `verifier_on_child_id: Option<usize>`
   - `verifier_off_child_id: Option<usize>`
11. Extended `CreateNode` event with `verifier_on: Option<bool>`:
   - root requires `verifier_on: None`
   - non-root requires `verifier_on: Some(bool)`
12. Implemented branch eligibility filtering in `determine_branching_node(...)`:
   - if sampled node already has both children, remove and resample
   - if no valid candidate remains, finalize rollout
13. Added max trajectory cap in branching stage:
   - `MAX_NUM_TRAJECTORIES = 16`
   - limit check occurs immediately after current leaf registration in `determine_branching_node(...)`
14. Added optional rule-based mode decision takeover:
   - new switch argument: `take_over_mode_decision`
   - extracted chosen-mode logic into dedicated `determine_chosen_mode(...)`
15. Populated per-node-step metadata in event replay application:
   - `Step.node_type` is set on `PlannerDecideNextStep`
   - `Step.step_finalized` is set on `PlannerEndStep` and terminal intervention tool responses
16. Hardcoded verifier-on first-expansion random split to `0.5` where branch type is chosen.

## Next steps

1. **Master-status transitions in rollout**
   - [x] When a trajectory ends, append the finished node to `leaf_node_ids`.
   - [x] Switch to `DeterminingBranchingNode` after `WorkingOnTrajectory` loop ends.
   - [x] Switch back to `WorkingOnTrajectory` in `determine_branching_node`.
   - [x] Continue branching until stop criterion (including max trajectory cap) is met.

2. **Branching-node selection logic**
   - [x] Implement weighted selection over trajectory nodes and enforce eligibility.
   - [x] Emit structural events in order when resuming work on selected parent:
      1) `CreateNode { parent_id: Some(selected_node_id), verifier_on: Some(...) }`
      2) `SetCurrentNode { node_id: new_child_id }`
   - [x] Defer concrete branch side (`verifier_on`/`verifier_off`) choice to working-trajectory phase.

3. **Event model extension (if needed)**
   - [ ] Decide whether to add explicit master-status events (e.g., `SetTreeMasterStatus`) for full replay fidelity.
   - [ ] If not adding events, document why derived-on-replay status is sufficient.

4. **Replay invariants**
   - [ ] Assert root creation occurs exactly once and first among structural events for fresh trees.
   - [ ] Assert `current_node_id.is_some()` whenever `AddAction` is applied.
   - [ ] Assert leaf bookkeeping consistency when trajectory end is detected.

5. **Browser / tooling updates**
   - [ ] Update browsing/debug flows to display `tree_master_status` and branching metadata.
