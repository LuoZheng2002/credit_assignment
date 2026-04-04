srun --pty --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=32 --ntasks=1 --mem=32G --account=bfdz-delta-gpu --time=00:30:00 \
    bash