# Evaluate one checkpoint of a ViT run.
#
#   bash run_eval_vit.sh                    # uses the settings below
#   MODE=gaze40 bash run_eval_vit.sh        # the 2026-08-25 gaze encoder (default)
#   MODE=phase bash run_eval_vit.sh         # the mask-supervised phase run
#   bash run_eval_vit.sh 120000             # a bare number overrides CKPT_STEP
#   bash run_eval_vit.sh --save_video       # extra flags are passed through
#
# MODE must match how the run was trained -- see run_learner_vit.sh. gaze loads
# no mask predictor, no gaze predictor and no pick classifier for the encoder,
# and the ViT is frozen; phase is the mask-supervised, phase-conditioned run.
#
# Recordings land in <RUN_DIR>/eval_recordings: raw_*.mp4/.webm,
# attention_*.mp4/.webm, episode_*.npz and a cumulative summary.json. Every
# invocation gets its own timestamped prefix, so repeated evals accumulate
# instead of overwriting. Afterwards:
#
#   python examples/analyze_eval.py --run_dir <RUN_DIR> \
#       --encoder_checkpoint <ENCODER_CKPT>

MODE="${MODE:-gaze40}"

# ============================ edit these ============================
N_TRAJS=5                   # episodes to run
if [ "$MODE" = "gaze40" ]; then
    CKPT_STEP=148000        # which checkpoint_<N> to evaluate
    RUN_NAME=2026-8-27_0_ball_pick_and_place_vit_gaze40_rl_run
    ENCODER_RUN=newgaze40_dil1
elif [ "$MODE" = "gaze" ]; then
    CKPT_STEP=166000
    RUN_NAME=2026-8-24_0_ball_pick_and_place_vit_gaze_rl_run
    ENCODER_RUN=tennis_ball_pick_and_place_vit_grounded_gaze
else
    CKPT_STEP=134000
    RUN_NAME=2026-8-20_1_ball_pick_and_place_vit_phase_rl_run
    ENCODER_RUN=tennis_ball_pick_and_place_vit_grounded_phase
fi
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_DIR="$SCRIPT_DIR/$RUN_NAME"

# A bare number as the first argument still overrides CKPT_STEP, so a quick
# one-off sweep does not mean editing the file. Anything else (a flag) is left
# alone and forwarded to train_rlpd.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    CKPT_STEP="$1"
    shift
fi

if [ ! -d "$RUN_DIR/checkpoint_$CKPT_STEP" ]; then
    echo "no such checkpoint: $RUN_DIR/checkpoint_$CKPT_STEP"
    echo "available:"
    ls -d "$RUN_DIR"/checkpoint_* 2>/dev/null | sed 's#.*/checkpoint_##' | sort -n | tr '\n' ' '
    echo
    exit 1
fi
echo "[$MODE] evaluating checkpoint_$CKPT_STEP of $RUN_NAME for $N_TRAJS episode(s)"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

if [ "$MODE" = "gaze40" ] || [ "$MODE" = "gaze" ]; then
    python "$ROOT/examples/train_rlpd.py" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-gaze \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/$ENCODER_RUN/best.msgpack" \
        --checkpoint_path="$RUN_DIR" \
        --actor_feature_overlay \
        --actor_feature_overlay_gamma=1.0 \
        --actor \
        --eval_checkpoint_step="$CKPT_STEP" \
        --eval_n_trajs="$N_TRAJS" \
        "$@"
else
    python "$ROOT/examples/train_rlpd.py" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/$ENCODER_RUN/best.msgpack" \
        --checkpoint_path="$RUN_DIR" \
        --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt" \
        --mask_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt" \
        --mask_selection_mode=pick_classifier \
        --pick_classifier_checkpoint_path="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick" \
        --actor_feature_overlay \
        --actor \
        --eval_checkpoint_step="$CKPT_STEP" \
        --eval_n_trajs="$N_TRAJS" \
        "$@"
fi
