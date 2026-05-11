Per step contribution to the final reward.

End conversation after tool response (can be done in the same forward pass), constant small learning rate

Not using tool call; incomplete; unfocused: each with a reward of 0 or 1, use GRPO to determine the advantage

Fatal error that causes trajectory to terminate midway: use GRPO to determine the advantage

Trajectory length score: should be taken care of by unfocused penalty.

Ratio: 5:1:1:1:1



How do we determine the importance of the training samples?


We need a smooth transition from all correct or all incorrect to half correct or half incorrect

The most direct way is to use a linear upslope and a linear downslope

Apart from the correctness ratio, we also need to consider the average trajectory length of the samples.

The target optimal trajectory length is 6. If way less or way more than this value, then we assign less importance to them.

This concludes the per step reward.

We also give rewards / penalties regarding the:
1. trajectory length (uniformly across a tree), first give a score, and then normalize, 20% importance
2. step quality (per step), probably useful in theory, but do not give a large weight, 15% importance total

This concludes all factors that calculate the advantage.

Plan for finding the final advantage:
1. Find the per-step mean and variance
2. Find the step length contribution
3. Find the step quality




Statistics to look for before and after training:
1. model proposed step quality
2. tree average trajectory length distribution
3. average accuracy (per leaf or per question)

