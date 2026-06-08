We move the sglang serve functionaltiy and model training functionality to the Modal computing vendor while maintaining the backward compatibility of being able to train on an HPC.

The training procedure is in src/orchestrator.rs:

We first do the inference to collect trajectories using sglang. The requests are dynamically determined by the Rust code.

Then we process and filter the trajectories in the Rust code.

Finally we train the model with the processed trajectories.

The interface of the Modal service should be:
1. During inference, the service acts as a transparent layer of sglang, receiving single requests and returning responses through network.
2. During training, the initial model is pulled from HuggingFace, then we upload the processed training trajectories for it to train. It constantly reports back the training progress through network. The trained model is stored on the Modal side for further inference and training.
3. We need a way to request the trained model through model cli name, config nickname and epoch number.