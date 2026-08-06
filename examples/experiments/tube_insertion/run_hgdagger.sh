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
            --exp_name=tube_insertion \
            --checkpoint_path=2026-07-25_tube_insertion_hgdagger \
            --demo_path="$ROOT/examples/bc_data/tube_insertion/tube_insertion*.pkl" \
            --pretrain_steps=2000
        ;;

    actor)
        python ../../train_hgdagger.py "$@" \
            --actor \
            --exp_name=tube_insertion \
            --checkpoint_path=2026-07-25_tube_insertion_hgdagger \
            --max_episode_steps=300 \
            --ip=localhost
        ;;

    eval)
        python ../../train_hgdagger.py "$@" \
            --actor \
            --exp_name=tube_insertion \
            --checkpoint_path=2026-07-25_tube_insertion_hgdagger \
            --eval_n_trajs=40 \
            --eval_checkpoint_step=230000 \
            --eval_checkpoint_step_interval=0 \
            --eval_max_episode_steps=300 \
            --save_video
        ;;
esac
