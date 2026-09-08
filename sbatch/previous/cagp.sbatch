#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o /mnt/lustre/work/hennig/hmx114/logs/%j_%x_%a.out
#SBATCH --array=0-29

# Define the base seed and calculate current seed from array task ID
BASE_SEED=42
SEED_MOD=$((SLURM_ARRAY_TASK_ID % 5))
DSET_MOD=$((SLURM_ARRAY_TASK_ID % 3))
SIGVAR=$((SLURM_ARRAY_TASK_ID % 2))
SEED=$((BASE_SEED + SEED_MOD))

if [[ $DSET_MOD == 0 ]]; then
   DSET="wine"
elif [[ $DSET_MOD == 1 ]]; then
   DSET="parkinsons"
else
   DSET="protein"
fi


if [[ $SIGVAR == 0 ]]; then
   TSIGVAR=(--trainable_output_scale)
else
   TSIGVAR=()
fi
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
export WANDB_API_KEY="$(cat ~/.wandb_key)"
export WANDB_USERNAME="julien-unruh"

echo "Starting job ${SLURM_ARRAY_JOB_ID}"
echo "Running on node: $(hostname)"

# Ensure the Slurm output directory exists
mkdir -p /mnt/lustre/work/hennig/hmx114/logs

# Change to repo directory and run with uv
cd ${REPO_DIR}
uv run python src/main.py --dataset "${DSET}" \
                        --split "0.7,0.2,0.1" \
                         --standardize "y,y,y,n,y,n" \
                         --gp "cagp" \
                         --kernel "RBFKeops" \
                         --likelyhood "gaussian" \
                         --mean "constant" \
                         --optimizer "adam" \
                         --learningrate 0.1 \
                         --iterations 500 \
                         --approximation_size 1024 \
                         --shuffle \
                         --seed "${SEED}" \
                         --device "cuda" \
                         "${TSIGVAR[@]}"


echo "Job ${SLURM_ARRAY_JOB_ID} completed"