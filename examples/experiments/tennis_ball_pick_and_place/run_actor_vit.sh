# ViT actor. MODE must match run_learner_vit.sh.
#
#   MODE=gaze3q  gaze drives everything the pick classifier used to, and the
#              policy no longer sees it. Four corrections over the 2026-09-02
#              run, which plateaued at a 38% pick intervention rate while its
#              place half beat the phase-one-hot run outright:
#
#              1. The gaze position is fed to the ViT's grounding query only.
#                 It used to also sit in the two state columns the policy MLP
#                 reads, where during pick it tracked the ball's image centroid
#                 to a 5.8px median error -- 82% of the ball's own 7px diameter
#                 -- against 0.84px from the mask branch. Two ball positions,
#                 one of them wrong by most of a ball. The basket is 55px, so
#                 the same error is 11% there, which is why place was unharmed.
#              2. The gaze->mask tie-break ranks candidates by distance, the
#                 rule the encoder's offline target already uses, instead of by
#                 the probability under the single gaze pixel -- values that
#                 measured 0.000 against 0.066 on the frames that flipped.
#              3. Selection is held until a new one repeats 3 frames. Measured
#                 switch rate 2.45% -> 0.75%, against 1.51% for the run that
#                 succeeded. Symmetric, so a dropped ball is not stranded in
#                 place.
#              4. Mask predictor retrained on the same 2026-08-25 sessions.
#                 The June-trained one segmented this ball at 0.515 IoU with
#                 0.54x its area -- accurate in position (0.80px) but half the
#                 size, so gaze fell outside it on 11% of frames. Retrained:
#                 0.949 IoU, 1.00x area, 2.6%. It also fixes the attention
#                 snapping onto the ball the moment it lands: the ball is
#                 completely occluded in the basket, and the old predictor
#                 still reported a 19px ghost there on 90.5% of those frames
#                 against 0.0% now. The encoder is unchanged -- retraining it
#                 was tried and the containment rule never fired, because a
#                 visible ball never overlaps the basket silhouette here.
#              discount 0.99, not 0.97: the reward is purely terminal and the
#              grasp lands at step 57 of 157, so at 0.97 the approach before it
#              saw 0.0086 of the reward.
#
#              5. discount 0.99 instead of 0.97. The reward is purely terminal
#                 -- robot_arm_penalty and grasp_penalty measure identically
#                 zero across both runs' buffers -- and the grasp lands at step
#                 57 of 157, so at 0.97 the approach that precedes it saw
#                 0.97^156 = 0.0086 of the reward. Since a successful grasp
#                 makes the place essentially automatic, that approach is
#                 exactly the part needing the credit; 0.99 lifts it to 0.21.
#                 Passed per-run, not set in TrainConfig, so phase and
#                 replicate keep the 0.97 they have to stay comparable at.
#
#              The pick classifier no longer selects the mask -- train_rlpd
#              passes its path only in pick_classifier mode. It is still loaded
#              as the gate on the reward: reward_func returns 0 for the whole
#              pick phase, and the gate only decides when the place classifier
#              starts being evaluated. That gate is load-bearing -- on the
#              2026-08-28 negative sets the place classifier clears its own 0.8
#              threshold on 12.8-38.5% of not-success frames, so without it an
#              empty hand passing over the basket ends the episode with a
#              spurious reward. It costs 0.71 ms per control step.
#
#   MODE=replicate  the same encoder as phase, into a FRESH run directory.
#              Tests whether the 2026-08-20b result reproduces at all: the
#              encoder file is byte-identical, so the only variables left are
#              RL's own randomness and the day's physical conditions. Needed
#              because every conclusion so far rests on single runs, and the
#              task measured brittle -- near-point success and failure differ
#              by 0.7px of ball position.
#   MODE=phase     ./run_actor_vit.sh    the 2026-08-20b run that succeeded (default)
#
# phase and replicate keep the June mask predictor and discount 0.97 on
# purpose: they are the baseline, and they have to stay comparable to what
# already ran.
#
# Modes for the runs that failed (nohand, gazemask, gaze40) were removed. Their
# encoders and checkpoints are still on disk -- see RUNS.md -- so re-adding a
# branch here is all it takes to evaluate one again.
MODE="${MODE:-phase}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

if [ "$MODE" = "gaze3q" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/gaze3q/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-09-02b_vit_gaze3q_noGazeState" \
        --mask_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt_0814/best.pt" \
        --mask_selection_mode=gaze \
        --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt_0825" \
        --actor_feature_overlay \
        --actor_feature_overlay_gamma=1 \
        --discount=0.99 \
        --actor
elif [ "$MODE" = "phase" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_phase/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-08-20b_vit_attnPlusHand_phase_maskobs_SUCCESS" \
        --mask_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt" \
        --mask_selection_mode=pick_classifier \
        --pick_classifier_checkpoint_path="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick" \
        --actor_feature_overlay \
        --actor_feature_overlay_gamma=1.0 \
        --actor
elif [ "$MODE" = "replicate" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_phase/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-09-02_vit_phase_replication" \
        --mask_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt" \
        --mask_selection_mode=pick_classifier \
        --pick_classifier_checkpoint_path="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick" \
        --actor_feature_overlay \
        --actor_feature_overlay_gamma=1.0 \
        --actor
else
    echo "unknown MODE='$MODE'. Use gaze3q, replicate or phase." >&2
    exit 1
fi
