The general formula for per-step advantage:

advantage = normalized_contribution * contribution_weight + average_trajectory_length_factor_normalized * trajectory_length_weight + (step_quality_factor1 + step_quality_factor2 + step_quality_factor3) * step_quality_weight

where

contribution_weight = 0.6

trajectory_length_weight = 0.25

step_quality_weight = 0.05

normalized contribution is calculated as following:
We first get contribution_mean_div_var from the current em fitting implementation.

Then we normalize this value within a tree to N(0, 1), applying both shifting and scaling.

After that normalization, we multiply by the win_loss_ratio_factor for that tree, and we get normalized_contribution.

win_loss_ratio_factor is based on the tree accuracy (correctness_ratio numerator / denominator):
- factor = 0.0 when accuracy is 0.0 or 1.0
- factor = 1.0 when accuracy is 0.5
- factor changes linearly between those points (equivalently: `1 - 2 * abs(accuracy - 0.5)`).

average_trajectory_length_factor_normalized is calculated as following:
For each tree, we find the average trajectory length across all trajectories (identified by leaf nodes), and apply a formula for finding the average trajectory length factor:
The target optimal trajectory length is 6.
For step length from 1 to 6, the formula is y = 1/5 * (x-1), so when x=1, y=0.0, and x=6, y=1.0. Then from x=6 to infty, there should be an exponential falloff starting from 1 and reaches 0.5 at x = 12.

Then we normalize the factor across all trees to N(0, 1).

step_quality_factor1 to step_quality_factor3 are the tool, complete and focused statistics in the StepQuality struct per step (use proper names during implementation). For each of them, if the value is true, assign a value of 1.0, otherwise 0.0. The values are normalized across all trees to N(0, 1).


Statistics to look for before and after training:
1. model proposed step quality
2. tree average trajectory length distribution
3. average accuracy (per leaf or per question)
