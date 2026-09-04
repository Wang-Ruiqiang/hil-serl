# Evaluate one checkpoint of a ViT run.
#
#   MODE=gaze3q bash run_eval_vit.sh        # the gaze run
#   MODE=phase  bash run_eval_vit.sh        # the 2026-08-20b run that succeeded (default)
#   bash run_eval_vit.sh 120000             # a bare number overrides CKPT_STEP
#   bash run_eval_vit.sh --eval_n_trajs=20  # extra flags are passed through
#
# MODE must match how the run was trained -- see run_learner_vit.sh. Modes for
# the runs that failed (nohand, gazemask, gaze40) were removed; their encoders
# and checkpoints are still on disk, see RUNS.md.
#
# Two kinds of video come out of an eval, and they are not the same thing:
#
#   <RUN_DIR>/eval_recordings/  raw_*.mp4/.webm is the 128x128 front_camera
#                               OBSERVATION the policy actually sees, plus
#                               attention_*.mp4/.webm (raw | attention |
#                               attention^gamma), episode_*.npz and a cumulative
#                               summary.json. Written by eval_recorder.py, on by
#                               default via --eval_record.
#   <RUN_DIR>/videos/<episode>/ the full-resolution camera feed, one mp4 per
#                               camera per episode, written by the env itself.
#                               This is the one to show people. On by default
#                               here; --nosave_video turns it off. FrankaEnv
#                               writes it to ./videos relative to the working
#                               directory, which is why the run is launched from
#                               inside RUN_DIR. It buffers a whole episode of
#                               full-res frames in RAM first, roughly 350 MB for
#                               a 200-step episode across two cameras.
#
# Every eval_recordings invocation gets its own timestamped prefix, so repeated
# evals accumulate instead of overwriting. Afterwards:
#
#   python examples/analyze_eval.py --run_dir <RUN_DIR> \
#       --encoder_checkpoint <ENCODER_CKPT>

MODE="${MODE:-phase}"

# ============================ edit these ============================
N_TRAJS=10                   # episodes to run
if [ "$MODE" = "gaze3q" ]; then
    CKPT_STEP=176000
    RUN_NAME=2026-09-02b_vit_gaze3q_noGazeState
    ENCODER_RUN=gaze3q
elif [ "$MODE" = "replicate" ]; then
    CKPT_STEP=134000
    RUN_NAME=2026-09-02_vit_phase_replication
    ENCODER_RUN=tennis_ball_pick_and_place_vit_grounded_phase
elif [ "$MODE" = "phase" ]; then
    CKPT_STEP=134000
    RUN_NAME=2026-08-20b_vit_attnPlusHand_phase_maskobs_SUCCESS
    ENCODER_RUN=tennis_ball_pick_and_place_vit_grounded_phase
else
    echo "unknown MODE='$MODE'. Use gaze3q, replicate or phase." >&2
    exit 1
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

# Must match the predictor the run trained with, or the masks the policy sees at
# eval are not the masks it learned on.
if [ "$MODE" = "gaze3q" ]; then
    MASK_PREDICTOR="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt_0814/best.pt"
else
    MASK_PREDICTOR="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt"
fi

# gazehybrid selects the mask from predicted gaze and loads no pick classifier;
# the other two select it from the classifier. Mixing them would feed the frozen
# grounding query the wrong two state columns.
if [ "$MODE" = "gaze3q" ]; then
    SELECTION=(--mask_selection_mode=gaze
               --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt_0825")
else
    SELECTION=(--mask_selection_mode=pick_classifier
               --pick_classifier_checkpoint_path="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick")
fi

# Launched from inside RUN_DIR so FrankaEnv's hard-coded ./videos lands with the
# run instead of wherever the shell happened to be. Every path handed to
# train_rlpd is absolute, so the working directory does not affect anything
# else. A subshell keeps the change local.
mkdir -p "$RUN_DIR/videos"
(
cd "$RUN_DIR" && python "$ROOT/examples/train_rlpd.py" \
    --exp_name=tennis_ball_pick_and_place \
    --encoder_type=vit-grounded \
    --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/$ENCODER_RUN/best.msgpack" \
    --checkpoint_path="$RUN_DIR" \
    --mask_predictor_checkpoint_path="$MASK_PREDICTOR" \
    "${SELECTION[@]}" \
    --actor_feature_overlay \
    --save_video \
    --actor \
    --eval_checkpoint_step="$CKPT_STEP" \
    --eval_n_trajs="$N_TRAJS" \
    "$@"
)

echo
echo "full-resolution video: $RUN_DIR/videos/<episode>/"
echo "observation + attention: $RUN_DIR/eval_recordings/"
