
sbatch \
    --job-name="qwen3_gpu4" \
    --output="slurm/qwen3_gpu4_%j.out" \
    --error="slurm/qwen3_gpu4_%j.err" \
    --gres=gpu:4 \
    slurm/gpu.slurm