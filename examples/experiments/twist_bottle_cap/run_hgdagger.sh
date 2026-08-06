#!/usr/bin/env bash
set -e

ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

MODE="${1:-}"
if [[ "$MODE" != "learner" && "$MODE" != "actor" && "$MODE" != "eval" ]]; then
    echo "Usage: bash run_hgdagger.sh learner|actor|eval [extra args]" >&2
    exit 1
fi
shift

case "$MODE" in
    learner)
        python ../../train_hgdagger.py "$@" \
            --learner \
            --exp_name=twist_bottle_cap \
            --checkpoint_path=2026-07-28_twist_bottle_cap_hgdagger \
            --demo_path="$ROOT/examples/bc_data/twist_bottle_cap/twist_bottle_cap*.pkl" \
            --pretrain_steps=2000 \
            --max_episode_steps=200 \
            --hand_action_weight=2.0
        ;;

    actor)
        python ../../train_hgdagger.py "$@" \
            --actor \
            --exp_name=twist_bottle_cap \
            --checkpoint_path=2026-07-28_twist_bottle_cap_hgdagger \
            --stage1_checkpoint_path=2026-07-28_lid_grip_hgdagger \
            --stage1_checkpoint_step=-1 \
            --max_episode_steps=200 \
            --ip=localhost
        ;;

    eval)
        python ../../train_hgdagger.py "$@" \
            --actor \
            --exp_name=twist_bottle_cap \
            --checkpoint_path=2026-07-28_twist_bottle_cap_hgdagger \
            --stage1_checkpoint_path=2026-07-28_lid_grip_hgdagger \
            --stage1_checkpoint_step=-1 \
            --eval_n_trajs=40 \
            --eval_checkpoint_step=-1 \
            --eval_checkpoint_step_interval=0 \
            --eval_max_episode_steps=400 \
            --save_video
        ;;
esac
