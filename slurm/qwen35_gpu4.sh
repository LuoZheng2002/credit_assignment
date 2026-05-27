
sbatch \
    --job-name="qwen35_gpu4" \
    --output="slurm/logs/qwen35_gpu4_%j.out" \
    --error="slurm/logs/qwen35_gpu4_%j.err" \
    --gres=gpu:4 \
    slurm/gpu.slurm