
sbatch \
    --job-name="qwen25_gpu1" \
    --output="slurm/qwen25_gpu1_%j.out" \
    --error="slurm/qwen25_gpu1_%j.err" \
    --gres=gpu:4 \
    slurm/gpu.slurm