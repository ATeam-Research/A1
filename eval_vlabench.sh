#!/bin/bash

# ============================================
# Load personal environment configuration
# ============================================
if [ -f "$PWD/.env.personal" ]; then
  echo "[env] Loading .env.personal"
  source "$PWD/.env.personal"
fi

# ============================================
# Activate Conda environment
# ============================================
if [ -n "$CONDA_ROOT" ] && [ -n "$CONDA_ENV" ]; then
  echo "[conda] Activating environment from $CONDA_ROOT: $CONDA_ENV"
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

python robot_experiments/vlabench/eval_client.py