A full model training procedure can be divided into the following steps:

1. We collect rollout logs through src/bin/bin_direct_tree.rs
2. We convert the rollout logs to training set through src/bin/bin_direct_training_set.rs (or src/bin/bin_browse_training_set.rs's side effect)
3. We run the training code in src_py/train folder to get the modified full model.
4. Use the modified model to do step 1, and repeat for num_epochs time.

Checkpoint system design:
1. We will run the full model training on different configurations, so each configuration should have its separate folder.
2. Each config's directory should be `results/[model_name]/[config_nickname]`
3. Inside each of these directories, there should be one or multiple `epoch_x` folders, where x denotes the epoch index, starting from 0.
4. Before the rollout collection begins, the initial base model should be loaded to the epoch_0 folder.
5. Then the rollout and training set generation will happen, and the two sqlite files will be generated in the epoch folder.
6. Then the python training code begins train the model in the epoch folder with the training set in the same epoch folder, and record the checkpoints folder in the epoch folder. The training code should be able to resume from the checkpoints folder under the same configuration.
7. When the training code finishes training for the current epoch, it will output the full model to the next epoch folder.
8. All the file and folder paths are determined by the orchestrator program, not by training or rollout programs to ensure consistency and easy modification.

