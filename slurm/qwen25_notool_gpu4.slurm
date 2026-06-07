#!/bin/bash

#SBATCH --job-name=qwen25_notool_gpu4
#SBATCH --output=slurm/logs/qwen25_notool_gpu4_%j.out
#SBATCH --error=slurm/logs/qwen25_notool_gpu4_%j.err
#SBATCH --gres=gpu:nvidia_a100:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=01:00:00

bash scripts/orchestrator/qwen25_notool_gpu4.sh
