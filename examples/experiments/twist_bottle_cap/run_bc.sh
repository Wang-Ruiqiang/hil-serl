#!/usr/bin/env bash
set -e

ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.7}"

RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
EXP_NAME="${EXP_NAME:-twist_bottle_cap}"
DEMO_PATH="${DEMO_PATH:-2026-02-02_keyboard_twist_bottle_cap_rl_run/demo_buffer}"
BC_CHECKPOINT_PATH="${BC_CHECKPOINT_PATH:-${RUN_DATE}_${EXP_NAME}_bc_demo_buffer}"
WANDB_PROJECT="${WANDB_PROJECT:-bc_hil_rl_comparison}"
WANDB_DESCRIPTION="${WANDB_DESCRIPTION:-${RUN_DATE}_${EXP_NAME}_bc_demo_buffer}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
ENABLE_TACTILE="${ENABLE_TACTILE:-1}"

python ../../train_bc.py "$@" \
    --exp_name="${EXP_NAME}" \
    --enable_tactile="${ENABLE_TACTILE}" \
    --bc_checkpoint_path="${BC_CHECKPOINT_PATH}" \
    --demo_path="${DEMO_PATH}" \
    --train_steps="${TRAIN_STEPS}" \
    --wandb_project="${WANDB_PROJECT}" \
    --wandb_description="${WANDB_DESCRIPTION}"
