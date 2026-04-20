# BROWSER TODO

## Scope and assumptions

1. No backward compatibility support for old jsonl schemas is required.
2. Browser replay should consider only `RolloutAction` sequence on a selected root-to-node path.
3. Browser replay should not reconstruct or mutate tree state during playback.
4. Node display should include only:
   - node id
   - an abbreviation of verifier/mode outcome
5. No edge metadata is required (only structural drawing with box characters).
6. No extra `Step` metadata panel is required; derive display from actions only.

## UI behavior

1. Level 1: question-level tree view
   - Show one tree for the selected question.
   - Render with box-drawing characters.
   - Root appears at left, leaves at right.
2. Level 2: node-level trajectory replay view
   - Selecting a node replays the concatenated action log from root to that node.
   - Replay is action-only (no tree operation replay).

## Node label format

For each node, render:

- `node_id`
- status abbreviation derived from actions in that node

Initial abbreviation plan:

- `VOFF`: verifier off (no verifier comment content)
- `VON`: verifier on and continue
- `VOW`: verifier on and overwrite-last-step
- `VOC`: verifier on and change-plan

Notes:

- Prefer deriving these from `RolloutAction` in the node action log.
- If action sequence is incomplete/inconsistent, use assertion to catch bug early.

## Data extraction plan

1. Load events and replay tree once using existing `Tree::apply_event`.
2. Build a browser view model from `Tree`:
   - node list with parent pointers
   - per-node label abbreviation derived from node actions
3. On node selection:
   - walk ancestors to root
   - concatenate each node's `action_log`
   - feed the resulting `Vec<RolloutAction>` to action replay UI

## Assertions to add before/while browser implementation

1. For every node index `i`, assert `tree.nodes[i].node_id == i`.
2. For non-root nodes, assert parent exists.
3. For each parent, assert child pointers (if present) point to existing nodes.
4. In action-derived abbreviation extraction:
   - assert there is exactly one `VerifierComment(...)` action per node step context
   - assert there is at most one `PlannerDecideNextStep(...)` per node step context
   - assert derived abbreviation is unambiguous

## Implementation steps

1. Add/prepare helper functions in browser code:
   - collect root-to-node action sequence
   - derive node abbreviation from node action log
2. Implement tree rendering with box characters and per-node labels.
3. Implement node selection interaction.
4. Implement action replay panel for selected node path.
5. Add assertions and fail-fast messages for inconsistent logs.
6. Manual validation on a few existing trajectory logs.

## Out of scope

1. Backward compatibility for old log schemas.
2. Tree-operation replay visualization.
3. Additional edge metadata rendering.
4. Extra non-action metadata panels.
