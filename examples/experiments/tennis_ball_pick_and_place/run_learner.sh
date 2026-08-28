SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python "$ROOT/examples/train_rlpd.py" "$@" \
    --exp_name=tennis_ball_pick_and_place \
    --checkpoint_path="$SCRIPT_DIR/2026-8-18_1_ball_pick_and_place_rl_run" \
    --demo_path="$ROOT/examples/demo_data/tennis_ball_pick_and_place_30_demos_2026-08-16.pkl" \
    --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt" \
    --mask_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt" \
    --mask_selection_mode=pick_classifier \
    --pick_classifier_checkpoint_path="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick" \
    --learner
