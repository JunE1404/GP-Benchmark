#!/bin/bash
#SBATCH --job-name=exact_gp_full_lbfgs_lsbound
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o /mnt/lustre/work/hennig/hmx114/logs/%j_%x_%a.out
#SBATCH --array=0-19

# Full-Wine exact GP on the LBFGS arm with a lengthscale lower bound. Both
# kernels are covered in one script: the array is 5 seeds x 2 output-scale
# settings x 2 kernels, so every configuration runs all five seeds (42-46).

BASE_SEED=42
SEED_MOD=$((SLURM_ARRAY_TASK_ID % 5))
SIGVAR=$((SLURM_ARRAY_TASK_ID / 5 % 2))
KERNEL_MOD=$((SLURM_ARRAY_TASK_ID / 10 % 2))
SEED=$((BASE_SEED + SEED_MOD))

if [[ $SIGVAR == 0 ]]; then
   TSIGVAR=(--trainable_output_scale)
else
   TSIGVAR=()
fi

if [[ $KERNEL_MOD == 0 ]]; then
   KERNEL="RBFKeops"
else
   KERNEL="matern32Keops"
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
uv run python src/main.py --dataset "wine" \
                        --split "0.9,0.0,0.1" \
                         --standardize "y,y,y,n,y,n" \
                         --gp "exact" \
                         --kernel "${KERNEL}" \
                         --likelyhood "gaussian" \
                         --mean "constant" \
                         --optimizer "lbfgs" \
                         --lgbfs_max_it 1 \
                         --iterations 500 \
                         --shuffle \
                         --seed "${SEED}" \
                         --device "cuda" \
                         --wandb \
                         --wandb_project "GP Test Runs" \
                         --wandb_entity "GP-Bench-Thesis" \
                         --float64 \
                         --deduplicate "first" \
                         --ls_bounds "1e-5" \
                         "${TSIGVAR[@]}"


echo "Job ${SLURM_ARRAY_JOB_ID} completed"
