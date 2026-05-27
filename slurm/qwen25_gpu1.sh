
sbatch \
    --job-name="qwen25_gpu1" \
    --output="slurm/logs/qwen25_gpu1_%j.out" \
    --error="slurm/logs/qwen25_gpu1_%j.err" \
    --gres=gpu:1 \
    slurm/gpu.slurm