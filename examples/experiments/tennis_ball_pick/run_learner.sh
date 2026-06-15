SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python "$ROOT/examples/train_rlpd.py" "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path="$SCRIPT_DIR/2026-6-15_0_ball_pick_rl_run" \
    --demo_path="$ROOT/examples/demo_data/tennis_ball_pick_20_demos_2026-06-15_16-39-46.pkl" \
    --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt" \
    --learner \
