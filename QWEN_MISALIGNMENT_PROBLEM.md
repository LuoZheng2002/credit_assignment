- Does not produce <end_step> properly when it wants step to end (can be solved by manually adding one, just performance consideration)
- Does not produce <tool_wait/> after a tool call (we can manually add it, affects performance mainly)
- Does not split the problem into steps of reasonable size.
- Does not use tool calls very often; tends to calculate things by hand.

- We need a new step to the agentic system that is specifically designed to make plans about steps
- If a direction (plan) is not promising, it will be displayed in a concise form for the planner to see.
- Is this sufficient or necessary?


- It's like monte carlo tree search, but it is more like inference-time self-correction.
- We have to make the model to continue even if the first plan does not work, and we need to inform the model about this.
- Apart from the plan of the initial attempt, we also need a concise description about why the initial attempt does not work, and what might be a possible new direction.

Curent step modes:
Append (continue, change plan)
OverwriteLastStep (continue, change plan)
Compact

Going to change to step modes:
Continue
OverwriteLastStep (intervention)
ChangePlan (intervention)

Post-step mandatory operations:
- Compact (compact steps of current attempt)
- UpdatePlan (update the plan when previous plan makes wrong assumptions or requires more information to continue, but the overall direction is promising)



We may need to force compact after each step is done.
step done -> verification and compact in parallel (implementation-wise compact first and then verify, but verify on uncompacted version) -> next step decision

(We not only try to make the model to produce the correct result in the first attempt, but also trains it to recover from mistakes)

We want most agentic interventions to be at mode choice level (continue or overwrite or update, etc.), but do not want to force the model to output contents in another style.

At the start of an attempt:
The model is prompted to generate a plan with explicit steps.

Don't know if explicit steps work for all problems.

Update plan seems to be necessary.

We need to make sure that the model has some chance to overwrite the last step (we can force it)

There will be no hint about whether the last step was overwritten and why it is the case for simplicity.

We need to make sure that the model has some chance to change the plan and decide to compact the conversation.

Step-level reward:

How does each trajectory diverge?

We assume that if verifier is introduced, there is more chance the trajectory succeeds; if we force non-trivial mode decisions (overwrite, update plan (maybe need to force unconditionally), change plan, compact (maybe need to force unconditionally)).



We dope these interventions to the trajectories so that at least 30% and at most 80% of the trajectories succeed.

(Collected from the same problem?)

After that, for successful trajectories, we remove a doped intervention at a time until the trajectory fails, and ...

For failed trajectories, we add one intervention at a time until the trajectory succeeds, and ...

What step to reward and penalize?
- In this way we can only make model to learn to bias towards non-trivial mode selection.
- We use more sophisticated context to collect oracle mode selection, and if a false positive is detected, we also force the mode.
- The major step to reward and penalize is the mode selection itself and the concrete step immediately after the mode selection.
- Changing a plan without verifier's comment might be ok, because we already prompted the model to change plan if the current direction is not promising.
- In this case, the step to reward and penalize is the decision to change plan and the step immediately follow after it.

- If we want to determine the step contribution at the beginning or in the middle, then we cannot do any intervention after the split point

- This means that in general we cannot have a lot of interventions to begin with
- If original trajectory fails and we add new interventions:
- it is expected that the new intervention is likely to make the new trajectory succeed
- to avoid confusion, we must add the intervention after the last intervention

- or ideally, we assume that the initial trajectory does not have intervention.

- If the original trajectory succeeds, then we need to remove an intervention, meaning that it needs to have at least one intervention to begin with.

This is a large restriction.

We can instead make all the steps after the split point to have interventions.

Interventions may not always contribute non-negatively and may confuse the model in some cases, but this is the best we can do.


We dope the interventions when generating the trajectory with some probability, and increase the probability until there are at least 30% of successful trajectories.

For each successful trajectory, we remove an intervention from the trajectory and make all subsequent steps to have full interventions.

This may not cause the new trajectory to fail.

We remove the intervention from right to left until we find a point where even after that point we have full interventions, the trajectory still fails.

We may not find such a point because we are adding more interventions in total.

Can we do binary search for this?

Not very possible ...

If our intervention oracle after the step without intervention saying that there is a mistake, or need to change plan, then the step without intervention is flawed.

The problem is whether the only step is responsible for the flaw, and whether the oracle is correct.

What's the difference from a verifier directly pointing out which step is flawed?

Our goal is to find a particular step to update, instead of needing to do the full rollout.

However, we cannot guarantee that the oracle is correct. We can make the intervention to only provide assessment information but does not force the step mode choices. Then we see if the trajectory fails.

There is still a problem regarding whether the particular step is the only step that causes failure.

We need to process the rest of the correct trajectory in the same way to do the comparison.

dynamic programming? -> reuse the trajectory as much as possible



For each failed trajectory, 


For each problem, we first do multiple rollouts, find the intervention probability that makes the accuracy at 30% - 70%.

Then we start from the beginning, for the first step, sample 4 trajectories with first step intervention and 4 without

If the average accuracy is below 50%, choose one with intervention, otherwise choose one without intervention, and keep moving.

For each step we have 8 samples, only choose the one that succeeds with intervention and fails without intervention, and calculate the baseline among all 8 samples. We need to assert that with intervention success rate > without intervention overall.

Early steps typically do not affect the result that much; later the intervention may be more and more important.

The failure modes are:
1. calculation mistakes
2. logical errors, conceptual errors, other mistakes
3. planning direction errors


Try <tool_wait> </tool_wait> for qwen 3 / 3.5

Make the compactor to spot if the final answer is found.

Limit the number of steps to 5.

Behavioral reward: tool calling, whether the current step only solves one step's problem; whether solved the step's problem

We find the correlation between a step and the final outcome

But this is expensive

Also, how do we sample the step? -> through verifier or overwrite step or change plan?

We hope that the step changes after the verifier intervenes and that the outcomes highly correlate with the step change.

We still need to measure the effect of verifier on the correctness of the dataset

How do we reuse the trajectories as much as possible?

We know that once the trajectory diverges at a point, then any parts after the diverge point cannot be compared.



Different trajectories ...



Add a search tool

step -> compact -> token generation is less than twice the ...
amortized

collecting trajectory

if we have many training data, then we can do dataset generation and then train on the dataset