#!/bin/bash

#SBATCH --job-name=mistral_grpo_notool
#SBATCH --output=slurm/logs/mistral_grpo_notool_%j.out
#SBATCH --error=slurm/logs/mistral_grpo_notool_%j.err
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=04:00:00

bash scripts/orchestrator/mistral_grpo_notool.sh
