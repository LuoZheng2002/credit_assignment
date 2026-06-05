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
#SBATCH --time=02:00:00

nohup bash scripts/orchestrator/qwen25_notool.sh > logs/nohup_qwen25_notool.txt 2>&1 &
sleep infinity