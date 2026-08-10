#!/bin/bash
#SBATCH --job-name=gp_setup
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=0:15:00
#SBATCH -o /mnt/lustre/work/hennig/hmx114/logs/%j_%x.out

set -euo pipefail

REPO_DIR="/mnt/lustre/work/hennig/hmx114/repos/gp-bench"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

cd "$REPO_DIR"
git rev-parse --short HEAD  # record the commit in the log

uv sync --frozen            # creates .venv + installs everything from uv.lock
echo "Setup complete"