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