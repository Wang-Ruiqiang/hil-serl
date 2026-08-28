# ViT learner. The resnet-pretrained baseline scripts are untouched:
# encoder_type is overridden on the command line, not in config.py.
#
# MODE=phase (default) is the mask-supervised, phase-conditioned encoder and is
# byte-identical to what this script ran before MODE existed.
# MODE=gaze is the gaze-supervised encoder: no mask predictor, no gaze
# predictor, no pick classifier, no phase one-hot. The ViT is frozen (the agent
# forces it -- CGL is off in that pipeline, so an unfrozen trunk would drift
# the grounding query's inputs with nothing to pull them back).
#
#   MODE=gaze ./run_learner_vit.sh
#
# MODE=gaze40 is the same pipeline with the encoder pretrained on the 40 demos
# recorded on 2026-08-25, whose eye-tracker calibration was corrected per
# session. It gets its own checkpoint directory on purpose: the readout is
# trained against one specific frozen trunk, so restoring a run trained on a
# different encoder would silently pair mismatched halves.
MODE="${MODE:-phase}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

# gaze40 uses demos exported from the same 2026-08-25 recordings its encoder was
# pretrained on, ten episodes from each of the three sessions so no single
# eye-tracker calibration dominates. The other modes keep the older export.
DEMO_PATH_GAZE40="$ROOT/examples/demo_data/tennis_ball_pick_and_place_new30_demos.pkl"
DEMO_PATH="$ROOT/examples/demo_data/tennis_ball_pick_and_place_30_demos_2026-08-16.pkl"

if [ "$MODE" = "gazemask" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-gaze \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/gazemask40_tactile/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-28_0_ball_pick_and_place_vit_gazemask_rl_run" \
        --demo_path="$DEMO_PATH_GAZE40" \
        --learner
elif [ "$MODE" = "gaze40" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-gaze \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/newgaze40_dil1/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-27_0_ball_pick_and_place_vit_gaze40_rl_run" \
        --demo_path="$DEMO_PATH_GAZE40" \
        --learner
elif [ "$MODE" = "gaze" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-gaze \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_gaze/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-24_0_ball_pick_and_place_vit_gaze_rl_run" \
        --demo_path="$DEMO_PATH" \
        --learner
else
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_phase/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-8-20_1_ball_pick_and_place_vit_phase_rl_run" \
        --demo_path="$DEMO_PATH" \
        --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt" \
        --mask_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt" \
        --mask_selection_mode=pick_classifier \
        --pick_classifier_checkpoint_path="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick" \
        --learner
fi
