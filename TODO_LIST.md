Currently we have finished the basic rollout code.

We target only 1 iteration for now.

We have 15000 samples.

We first use static dataset generation.

We've done the dataset generation, and we currently use a proportion cutoff.

Maybe we need to have a cumulative curve.

We need to collect the first 15000 * 16 samples, or a fixed gpu hour?

Validation set?

Maybe the first 5000 in-distribution but validation samples is a good way to verify the accuracy increase. (temperature = 0)

Maybe we do 3 iterations and collect the 

TODO:
1. Training set generation (fixed size?) We do full 15000 samples * 16 (probably)
2. Training code
3. Ablation study
4. TBD

We want a fully automated pipeline:

We do the following in one epoch:
1. We assume that a model is at the corresponding checkpoint position (Checkpoint 0 should have a model BEFORE the 0th epoch training)
2. Do the validation and collect accuracy (statistics: validation accuracy, etc.)
3. Collect training set through epochs with temperature = 0.7
4. Run the training code (if this is not the last checkpoint), and put the trained model to the next checkpoint


We first need to build the components before orchestrating them.


Going to add model answer, correct answer and judgment metadata to training trajectories, since there might be failure modes that can be caught, like \boxed{answer}.


TODO: add a config field purpose: Purpose (Training, Validation, Testing) to rollout trajectories; 
TODO: add potential ablation variants configuration, like which strategy to use for calculating advantage, branching strategy, etc.



We need AI to help us to read the training trajectories, and see where the problem lies.

We need automatic downloading and inspecting. Only download if the corresponding file does not exist.

