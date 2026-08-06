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
            --exp_name=tennis_ball_pick \
            --checkpoint_path=2026-07-23_1_tennis_ball_pick_hgdagger \
            --demo_path="$ROOT/examples/bc_data/tennis_ball_pick/tennis_ball_pick*.pkl" \
            --pretrain_steps=5000 \
            --hand_action_weight=2.0
        ;;

    actor)
        python ../../train_hgdagger.py "$@" \
            --actor \
            --exp_name=tennis_ball_pick \
            --checkpoint_path=2026-07-23_1_tennis_ball_pick_hgdagger \
            --stage1_checkpoint_path=2026-07-23_1_tennis_ball_pick_hgdagger \
            --stage1_checkpoint_step=70000 \
            --max_episode_steps=150 \
            --ip=localhost
        ;;

    eval)
        python ../../train_hgdagger.py "$@" \
            --actor \
            --exp_name=tennis_ball_place \
            --checkpoint_path=2026-07-23_1_tennis_ball_place_hgdagger \
            --stage1_checkpoint_path=2026-07-23_1_tennis_ball_pick_hgdagger \
            --stage1_checkpoint_step=70000 \
            --eval_n_trajs=30 \
            --eval_checkpoint_step=140000 \
            --eval_checkpoint_step_interval=0 \
            --eval_max_episode_steps=600 \
            --save_video
        ;;
esac
