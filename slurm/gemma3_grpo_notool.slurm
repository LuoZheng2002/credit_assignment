#!/bin/bash

#SBATCH --job-name=gemma3_grpo_notool
#SBATCH --output=slurm/logs/gemma3_grpo_notool_%j.out
#SBATCH --error=slurm/logs/gemma3_grpo_notool_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=04:00:00

bash scripts/orchestrator/gemma3_grpo_notool.sh