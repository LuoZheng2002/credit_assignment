This project is about verifying a fine-grained credit assignment strategy for agentic LLM.

It makes the LLM to break a problem into multiple steps and solve it step by step.

We then make the LLM to branch at the middle of the problem solving with some perturbations to lead to different outcomes.

Then we assign the credits to the steps based on the different outcomes.

If the outcome diverges at the very end, then the diverged steps at the end are very likely responsible.

If the average outcome given an early step is above average, then the early step is likely to be good, but with less confidence.

We model each step's contribution to be a scalar either positive or negative that adds to the final outcome. If the final outcome is positive, then it succeeds, otherwise it ails.

We model the prior of the steps to be a gaussian distribution, and then use expectation maximization algorithm to fit the different outcomes to get each step's posterior. This allows us to naturally quantify the "contribution direction" and "confidence" for each step, and use this information to determine the advantage of each step.

## Submodule setup

This repository depends on the `research-utility` submodule at `research-utility/`.

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned, initialize and update submodules:

```bash
git submodule update --init --recursive
```

To update `research-utility` to the latest remote commit and record that pointer in this repo:

```bash
git submodule update --remote --merge research-utility
```
