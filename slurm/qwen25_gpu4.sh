
sbatch \
    --job-name="qwen25_gpu4" \
    --output="slurm/logs/qwen25_gpu4_%j.out" \
    --error="slurm/logs/qwen25_gpu4_%j.err" \
    --gres=gpu:4 \
    slurm/gpu.slurm