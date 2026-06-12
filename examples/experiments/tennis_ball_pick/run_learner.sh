ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=2026-1-15_0_ball_place_ablation_keyboard_rl_run \
    --demo_path=../../demo_data/tennis_ball_place_ablation_25_demos_2026-01-15_14-46-40.pkl \
    --learner \