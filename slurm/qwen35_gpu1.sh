
sbatch \
    --job-name="qwen35_gpu1" \
    --output="slurm/qwen35_gpu1_%j.out" \
    --error="slurm/qwen35_gpu1_%j.err" \
    --gres=gpu:1 \
    slurm/gpu.slurm