#!/bin/bash
# #SBATCH --job-name=empty       # Job name
# #SBATCH --output=slurm/empty_%j.out        # Standard output log (%j = job ID)
# #SBATCH --error=slurm/empty_%j.err         # Error log
#SBATCH --partition=gpuA100x4
# #SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=02:00:00

sleep infinity