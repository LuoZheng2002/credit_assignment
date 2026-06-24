We use qwen2.5-7b for all experiments, including the standard one (with 4 trunks and 16 leaves, tool enabled), the no-tool variant, the grpo variant (8 trunks and 8 leaves), the notool-grpo variants, 8 leaves and 32 leaves variants, and "tempo" and "treerpo" ablation studies (see scripts/orchestrator_modal).

For qwen3-4b, we only consider the permutations between tree vs. grpo, tool vs. no-tool. This gives 4 experiments.

For gemma3, llama3.1 and mistral, we only consider tree vs. grpo, both without tool.

For experiment result display, we should display two main table:
1. The main table should display each model's performance when not fine-tuned, using grpo, and using our tree method. Ideally, the trend should be not-fine-tuned < grpo < our tree method regarding accuracy. The columns should be datasets. We have 3 in-distribution datasets and 4 out-of-distribution datasets. The columns should be models and training setups. Each model's three training methods (not fine-tuned, using grpo, and using our tree method) should be grouped together. For qwen models, the setup of using tool and not using tool should be displayed as if tool-using qwen models and not-tool-using qwen models are independent models.
2. The ablation table should be similar to the main table, and we replace the grpo with "tempo" an "treerpo" setups.

Finally, we want to have a line chart regarding how the number of tree leaves (how much we branch) in a tree affects the trained model performance. It is expected to be a concave shape where there is a sweet spot regarding the number.

