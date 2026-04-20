Currently we have a Tree with nodes (with priors based on its verifier and mode characteristic) that contribute to the trajectories and lead to a binary success/ failure outcome.

We assume that the hidden outcome is a continuous scalar that initially starts at 0 and each node in the way shifts it either positively or negatively and the effects are combined through signed addition, and if the final outcome is > 0 then the observable outcome is success, otherwise failure.

I'd like to model this as a maximum a posteriori EM with binary threshold likelihood.

A normal node (VerifierAndModeSummary::VerifierOff) has a prior with mean 0 and std 1.
Special nodes (other variants of VerifierAndModeSummary) has a prior with mean \mu_{special_mode_name} and std 1.
Each \mu_{special_mode_name} has a prior with mean 0 and std 1. (Don't know if std=1 is proper here).

