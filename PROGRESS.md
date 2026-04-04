We make all model fowarding / training calls on the python side. Forwarding type includes: vllm and api for forward, that does not need python end; and model logit collection and backward, which is achieved through DeepSpeed ZeRO-3.

In the training phase, the main logic is on the Rust side, that simulates a multi-agent environment, and then send request and receive response from the python side.