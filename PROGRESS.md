We make all model fowarding / training calls on the python side. Forwarding type includes: vllm and api for forward, that does not need python end; and model logit collection and backward, which is achieved through DeepSpeed ZeRO-3.

In the training phase, the main logic is on the Rust side, that simulates a multi-agent environment, and then send request and receive response from the python side.

Do we want sub-tasks?

Calculation; program running; 

verify if the result is correct?

planner; executor; verifier; 

context ()

Assumption: verifier can identify many different kinds of mistakes

Planner can declare different methods:
If verifier has spoken: respond to verifier. For simplicity, we assume that verifier is always correct.

The act of changing plan should only be adviced by the verifier
1. Change plan (needs to be explicitly stated )

2. Keep going on the current plan.

3. Rewrite the last step. (actual trajectory vs. trajectory seen by models and used for training)
This overwrites the last ...

4. Compact (Compact results)



Tool calls:



Numerical calculation;



tool call...