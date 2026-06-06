#!/bin/bash

#SBATCH --job-name=gemma3_notool
#SBATCH --output=slurm/logs/gemma3_notool_%j.out
#SBATCH --error=slurm/logs/gemma3_notool_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --account=bfdz-delta-gpu
#SBATCH --time=04:00:00

max_retries=5
for trial in $(seq 1 "$max_retries"); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Trial ${trial}/${max_retries}: running gemma3_notool orchestrator"
    trial_start_time=$(date +%s)
    if bash scripts/orchestrator/gemma3_notool.sh; then
        trial_end_time=$(date +%s)
        elapsed_seconds=$((trial_end_time - trial_start_time))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Trial ${trial}/${max_retries}: succeeded (elapsed ${elapsed_seconds}s)"
        exit 0
    fi
    trial_end_time=$(date +%s)
    elapsed_seconds=$((trial_end_time - trial_start_time))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Trial ${trial}/${max_retries}: failed (elapsed ${elapsed_seconds}s)"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All ${max_retries} trials failed"
exit 1
