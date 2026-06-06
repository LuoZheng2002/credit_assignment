if [ $# -ne 1 ]; then
  echo "Usage: . srun.sh <jobid>" >&2
  return 1
fi

srun --overlap --jobid "$1" --pty bash