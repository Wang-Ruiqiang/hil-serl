#!/usr/bin/env bash
set -e

ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.7}"

RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
RUN_LABEL_DATE="${RUN_LABEL_DATE:-$(date +%-m-%-d)}"
EXP_NAME="${EXP_NAME:-tennis_ball_place}"
DEMO_PATH="${DEMO_PATH:-$ROOT/examples/bc_data/tennis_ball_place}"
# BC_CHECKPOINT_PATH="${BC_CHECKPOINT_PATH:-${RUN_DATE}_${EXP_NAME}_bc_demo_buffer}"
BC_CHECKPOINT_PATH="2026-07-23_tennis_ball_place_bc"
WANDB_DESCRIPTION="${WANDB_DESCRIPTION:-bc-${EXP_NAME}-${RUN_LABEL_DATE}}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:--1}"
ENABLE_TACTILE="${ENABLE_TACTILE:-1}"

python ../../train_bc.py "$@" \
    --exp_name="${EXP_NAME}" \
    --enable_tactile="${ENABLE_TACTILE}" \
    --bc_checkpoint_path="${BC_CHECKPOINT_PATH}" \
    --demo_path="${DEMO_PATH}" \
    --train_steps="${TRAIN_STEPS}" \
    --checkpoint_period="${CHECKPOINT_PERIOD}" \
    --wandb_description="${WANDB_DESCRIPTION}"
