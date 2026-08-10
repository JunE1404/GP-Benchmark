#!/bin/bash
#SBATCH --job-name=dockstring
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH -o /mnt/lustre/work/hennig/hmx114/logs/%j_%x_%a.out

# Define the base seed and calculate current seed from array task ID
BASE_SEED=42
SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

# Define paths
REPO_DIR="/mnt/lustre/work/hennig/hmx114/repos/gp-bench"

# Set CUDA paths for KeOps (some of this stuff might not be necessary)
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LIBRARY_PATH
export KEOPS_CACHE_DIR=/tmp/keops_cache

# Think this is for downloading UCI data
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

# Set W&B credentials
export WANDB_API_KEY="foo"
export WANDB_USERNAME="bar"

echo "Starting job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Using seed: ${SEED}"
echo "Running on node: $(hostname)"

# Change to repo directory and run with uv
cd ${REPO_DIR}
uv run python scripts/main.py \
    --approximation-type=cagp \
    --seed=${SEED} \
    --n-iters=1000 \
    --optimizer=adam \
    --lr=0.01 \
    --subset-size=2048 \
    --train-gram-tile-size=64 \
    --test-train-gram-chunk-size=256 \
    --log-wandb


echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} completed"