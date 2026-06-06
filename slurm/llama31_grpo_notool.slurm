#!/bin/bash

#SBATCH --job-name=llama31_grpo_notool
#SBATCH --output=slurm/logs/llama31_grpo_notool_%j.out
#SBATCH --error=slurm/logs/llama31_grpo_notool_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=04:00:00

bash scripts/orchestrator/llama31_grpo_notool.sh