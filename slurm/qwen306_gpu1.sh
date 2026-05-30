sbatch \
    --job-name="qwen306_gpu1" \
    --output="slurm/logs/qwen306_gpu1_%j.out" \
    --error="slurm/logs/qwen306_gpu1_%j.err" \
    --gres=gpu:1 \
    slurm/gpu.slurm