#!/bin/bash
#SBATCH --job-name=empty       # Job name
#SBATCH --output=slurm/empty_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=slurm/empty_%j.err         # Error log
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=03:00:00

sleep infinity
