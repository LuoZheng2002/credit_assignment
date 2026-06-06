
sbatch \
    --job-name="gpu1" \
    --output="slurm/logs/gpu1_%j.out" \
    --error="slurm/logs/gpu1_%j.err" \
    --gres=gpu:1 \
    slurm/gpu.slurm