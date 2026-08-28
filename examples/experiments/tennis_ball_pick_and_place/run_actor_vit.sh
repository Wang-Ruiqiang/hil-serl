# ViT actor. See run_learner_vit.sh for what MODE selects; the two must match.
# MODE: phase (default) | gazemask | gaze | gaze40
MODE="${MODE:-phase}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

if [ "$MODE" = "gazemask" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-gaze \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/gazemask40_tactile/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-28_0_ball_pick_and_place_vit_gazemask_rl_run" \
        --actor_feature_overlay \
        --actor_feature_overlay_gamma=1.0 \
        --actor
elif [ "$MODE" = "gaze40" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-gaze \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/newgaze40_dil1/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-27_0_ball_pick_and_place_vit_gaze40_rl_run" \
        --actor_feature_overlay \
        --actor_feature_overlay_gamma=1.0 \
        --actor
elif [ "$MODE" = "gaze" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-gaze \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_gaze/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-24_0_ball_pick_and_place_vit_gaze_rl_run" \
        --actor_feature_overlay \
        --actor_feature_overlay_gamma=1.0 \
        --actor
else
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_phase/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-20_1_ball_pick_and_place_vit_phase_rl_run" \
        --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt" \
        --mask_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt" \
        --mask_selection_mode=pick_classifier \
        --pick_classifier_checkpoint_path="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick" \
        --actor_feature_overlay \
        --actor
fi
