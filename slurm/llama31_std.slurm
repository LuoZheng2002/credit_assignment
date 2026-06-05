#!/bin/bash

#SBATCH --job-name=llama31_std
#SBATCH --output=slurm/logs/llama31_std_%j.out
#SBATCH --error=slurm/logs/llama31_std_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=04:00:00

nohup bash scripts/orchestrator/llama31_std.sh > logs/nohup_llama31_std.txt 2>&1 &
sleep infinity