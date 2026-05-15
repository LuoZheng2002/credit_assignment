# TreePO Summary (Methodology + Implementation)

## Paper identity

- **Title:** TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling
- **Core goal:** Replace independent sequential rollouts with **segment-level tree sampling** to improve RL training efficiency and exploration, then use tree-aware advantage estimation.

## Methodology

1. **Core observation and reformulation**
   - Multiple sampled reasoning trajectories often share long token prefixes.
   - Standard i.i.d. rollouts recompute these shared prefixes repeatedly (wasted KV cache compute).
   - TreePO treats generation as a tree search with shared-prefix reuse.

2. **Segment-level tree rollout algorithm**
   - Decode in fixed-length segments (`l`) rather than one full trajectory at once.
   - Maintain a queue of active prompts (tree nodes).
   - At each step:
     - branch active nodes according to branching policy,
     - run inference on active nodes,
     - stop finished/failed nodes,
     - append unfinished outputs back to queue,
     - apply fallback when needed to meet trajectory budget.
   - Tree dimensions are described via width (`w` trajectories), depth (`d` segments), segment length (`l`), branching budget (`b`).

3. **Heuristic control policies**
   - **Early stop** for flawed/repetitive generations.
   - **Branching budget transfer** to rebalance active paths when some terminate early.
   - **Depth-first fallback** only when no active path remains and width target is unmet.
   - Optional probability-based branching assignments from segment logprobs.

4. **Tree-based advantage estimation**
   - Starts from GRPO/DAPO-like token-level objective.
   - Build nested subgroups for each trajectory based on shared predecessor nodes at each depth.
   - For each subgroup level, compute relative reward to subgroup mean; aggregate across levels.
   - Uses global variance normalization and rejection of all-correct/all-wrong full groups (DAPO-style condition).

## Implementation details (what to reproduce)

### Training objective and estimator

- Objective follows DAPO-modified GRPO clipping/token-level style.
- Trajectory reward is used to compute subgroup-relative advantages across tree levels.
- Final TreePO advantage is aggregated over depth-wise subgroup advantages, then normalized.

### Data and model setup

- **Student model:** Qwen2.5-7B base (main RL runs).
- Additional efficiency tests on Qwen2.5-7B-Instruct and Qwen2.5-Math-7B-Instruct.
- **Training data:** MATH subset (difficulty 3-5, ~8k) + DeepScaler samples (~40k).
- **Evaluation:** AIME24, AMC23, MATH500, MINERVA, OlympiadBench; majority-vote metrics.

### Tree sampling hyperparameters

- Explored `d x l` settings under fixed max response budget, including:
  - `28 x 256`,
  - `14 x 512`,
  - `7 x 1024`.
- Typical max width `w = 16` (same group-size budget as sequential baseline).
- Branching schedule example: binary budget by depth (`2^d`) under their notation.
- Two training modes:
  - **Fixed Init Divergence**,
  - **More Init Divergence** (random initial divergence budget, e.g., 2-8).

### System/runtime stack

- Training framework: VeRL with FSDP mode.
- Inference backend: vLLM (V0 engine).
- Hardware in training section: 64 GPUs.
- Reported training knobs include LR around `1e-6`, warmup 10 steps, batch size 512, max 20 epochs, checkpoint interval 50 steps.

### Efficiency benchmark protocol (paper)

- Dedicated offline throughput study on H100 80GB.
- No tensor/data parallel in that benchmark; ~60% utilization target.
- Metrics:
  - token throughput (TokenPS),
  - trajectory throughput (TrajPS).
- Fixed per-trajectory budget (`B=7000` tokens) while varying depth/segment combinations and rollout count.

## Reported findings

1. **Efficiency**
   - Tree-based sampling reports substantial GPU-hour reductions (paper reports ranges around ~22%-43% depending on setting).
   - Offline throughput gains reported around +30% TokenPS and +40% TrajPS on average.

2. **Performance**
   - Tree sampling improves over GRPO baseline in their major@16 evaluation table.
   - Adding TreePO estimator on top of tree sampling improves stability and overall metrics further in their experiments.

3. **Ablation insights**
   - Intermediate depth/segment trade-off works best (reported sweet spot near `14 x 512` in their runs).
   - Misaligned fallback (segment mismatch) hurts accuracy and increases response length.
   - Static probability-based branching heuristics (always favor low/high prob paths) did not help; monotonic controls can be harmful.

## Practical takeaway

- TreePO emphasizes **systems-aware RL sampling**: shared-prefix reuse + heuristic branching/fallback to cut sampling cost.
- Its optimization contribution is a **tree-level subgroup advantage** that tries to provide finer credit than flat GRPO grouping.
- Main value proposition is the cost/performance trade-off, especially where rollout generation dominates training compute.
