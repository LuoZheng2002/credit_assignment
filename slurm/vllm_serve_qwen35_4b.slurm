#!/bin/bash
#SBATCH --job-name=qwen35_4b_vllm_serve       # Job name
#SBATCH --output=slurm/qwen35_4b_vllm_serve_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=slurm/qwen35_4b_vllm_serve_%j.err         # Error log
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=00:45:00

source activate_environment.sh


vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8002