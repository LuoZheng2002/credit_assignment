#!/bin/bash
#SBATCH --job-name=qwen3_4b_vllm_serve_gpu4       # Job name
#SBATCH --output=slurm/qwen3_4b_vllm_serve_gpu4_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=slurm/qwen3_4b_vllm_serve_gpu4_%j.err         # Error log
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=00:45:00

source activate_environment.sh

vllm serve Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 4
