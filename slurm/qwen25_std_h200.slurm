#!/bin/bash

#SBATCH --job-name=qwen25_std
#SBATCH --output=slurm/logs/qwen25_std_%j.out
#SBATCH --error=slurm/logs/qwen25_std_%j.err
# #SBATCH --gres=gpu:nvidia_a100:1
# #SBATCH --gres=gpu:nvidia_a100:5
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:h200:1
# #SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
# #SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=04:00:00

bash scripts/orchestrator/qwen25_std.sh
