
sbatch \
    --job-name="qwen3_gpu1" \
    --output="slurm/qwen3_gpu1_%j.out" \
    --error="slurm/qwen3_gpu1_%j.err" \
    --gres=gpu:1 \
    slurm/gpu.slurm