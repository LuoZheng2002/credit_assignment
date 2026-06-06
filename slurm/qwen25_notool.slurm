#!/bin/bash

#SBATCH --job-name=qwen25_notool
#SBATCH --output=slurm/logs/qwen25_notool_%j.out
#SBATCH --error=slurm/logs/qwen25_notool_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=04:00:00

bash scripts/orchestrator/qwen25_notool.sh