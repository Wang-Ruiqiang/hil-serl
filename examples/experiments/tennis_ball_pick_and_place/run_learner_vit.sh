# ViT learner. MODE must match run_actor_vit.sh.
#
#   MODE=gaze3q  gaze drives everything the pick classifier used to do.
#              Pretraining: gaze inside the dilated ball selects the ball's own
#              silhouette as the attention target, gaze inside the basket selects
#              the basket, and gaze on neither -- 51.8% of frames, sitting a
#              median 9.1px off the ball, i.e. on the fingers closing around it --
#              is supervised as a blob at the gaze point itself. The grounding
#              query is conditioned on the gaze position, so the question asked
#              and the answer taught come from one signal. The 2026-08-28 tactile
#              run failed exactly there: 14.7% of its frames held the ball while
#              the sensor read nothing, so the conditioner could never commit.
#              At RL time a gaze predictor supplies the position. On held-out
#              episodes it lands a median 8.7px from the real fixation and
#              reproduces the ball/basket/neither choice 95.9% of the time.
#              discount 0.99 instead of 0.97: the reward is purely terminal --
#              robot_arm_penalty and grasp_penalty measure identically zero in
#              both runs' buffers -- and the grasp lands at step 57 of 157, so
#              at 0.97 the approach before it saw 0.97^156 = 0.0086 of the
#              reward. A successful grasp makes the place essentially
#              automatic, so that approach is exactly the part needing the
#              credit; 0.99 lifts it to 0.21. Passed per-run rather than set in
#              TrainConfig, so phase and replicate keep the 0.97 they have to
#              stay comparable at.
#
#              The pick classifier no longer selects the mask. It stays loaded
#              as the reward gate -- reward_func returns 0 through the whole
#              pick phase, and the gate only decides when the place classifier
#              starts being evaluated. Removing it was tried and reverted: on
#              the 2026-08-28 negative sets the place classifier clears its own
#              0.8 threshold on 12.8-38.5% of not-success frames, so an empty
#              hand passing over the basket would end the episode with a
#              spurious reward. Cost is 0.71 ms per control step.
#
#   MODE=replicate  the same encoder as phase, into a FRESH run directory.
#              Tests whether the 2026-08-20b result reproduces at all: the
#              encoder file is byte-identical, so the only variables left are
#              RL's own randomness and the day's physical conditions. Needed
#              because every conclusion so far rests on single runs, and the
#              task measured brittle -- near-point success and failure differ
#              by 0.7px of ball position.
#
#   MODE=phase  the 2026-08-20b configuration that succeeded (default).
#
# In every mode the ViT trunk and the grounding query are frozen; RL trains the
# spatial readout and the bottleneck. Verified by diffing 08-20b's RL checkpoint
# against its pretraining msgpack: 83 of 88 tensors byte-identical.
MODE="${MODE:-phase}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

# Each encoder pairs with demos from the recordings it was pretrained on.
DEMO_0814="$ROOT/examples/demo_data/tennis_ball_pick_and_place_30_demos_2026-08-16.pkl"
# gaze3q only. NOT the as-recorded new30 file: that one was written without a
# working mask predictor, so front_camera_mask, _mask1 and _mask2 are empty in
# all 5822 transitions, and with no mask the selected index never moved, leaving
# a constant [1.0, 0.0] in the two columns the grounding query reads. RLPD draws
# half of every batch from the demo buffer, so half of every update fed the mask
# CNN a blank image and told a gaze-conditioned encoder the operator was staring
# at the top-right corner. (The 2026-08-16 demos the successful run used are
# intact by comparison, 0.0% / 2.8% / 0.0% empty, and predate the 2-wide slot:
# they are 11 columns, which prepare_replay_transition truncates from the right,
# leaving a correct and varying [pick, place].)
#
# This file is re-exported straight from the recordings with
# --state_gaze_slot=gaze_xy, so the slot carries the eye tracker's own fixation
# -- gaze_uv_in_realsense normalised over the full 640x480 frame, the same
# convention train_encoder.py conditioned the query on and the same one
# gaze_xy_norm_from_heatmap produces online, since the env resizes the whole
# frame rather than cropping it. 7853 transitions over 39 episodes, gaze
# continuous over 7848 distinct positions instead of the predictor's 30, and
# masks read from the recordings' own SAM output: 14.2% / 32.7% / 0.0% empty,
# selecting the basket on 32.2% of frames against 31.6% in the buffer.
DEMO_0825="$ROOT/examples/demo_data/tennis_ball_pick_and_place_new30_demos_realgaze_mp0814.pkl"
MASK_PREDICTOR="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt"
# gaze3q only: trained on the 2026-08-14 sessions, the ones with hand-drawn
# ball and basket masks. Measured on a held-out slice of those manual labels
# against the June predictor and one trained on 2026-08-25's SAM3 output:
#   ball IoU 0.957 / 0.905 / 0.596, miss 1.0% / 6.6% / 21.7%,
#   centroid 0.12 / 0.24 / 0.87 px, basket IoU 0.998 / 0.984 / 0.980.
# SAM3 is why the 08-25 one is unusable: on identical frames the hand-drawn
# ball mask is empty 1.8-3.4% of the time and SAM3's is empty 27-31%, so a
# model fitted to it learns to drop the ball. On 2026-08-25 -- unseen by this
# checkpoint -- the three lose the ball on 1.4% / 13.1% / 34.7% of frames and
# flicker present/absent on 1.79% / 6.66% / 7.14% of steps.
# phase and replicate keep the June one so they stay comparable to the runs
# already on disk.
MASK_PREDICTOR_0814="$ROOT/examples/gaze_data_process/SAM_process/mask_predictor_ckpt_0814/best.pt"
PICK_CLF="$ROOT/examples/reward_classifier/classifier_ckpt_ball_pick"

if [ "$MODE" = "gaze3q" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/gaze3q/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-09-02b_vit_gaze3q_noGazeState" \
        --demo_path="$DEMO_0825" \
        --mask_predictor_checkpoint_path="$MASK_PREDICTOR_0814" \
        --mask_selection_mode=gaze \
        --gaze_predictor_checkpoint_path="$ROOT/examples/gaze_data_process/gaze_heatmap_ckpt_0825" \
        --discount=0.99 \
        --learner
elif [ "$MODE" = "replicate" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_phase/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-09-02_vit_phase_replication" \
        --demo_path="$DEMO_0814" \
        --mask_predictor_checkpoint_path="$MASK_PREDICTOR" \
        --mask_selection_mode=pick_classifier \
        --pick_classifier_checkpoint_path="$PICK_CLF" \
        --learner
elif [ "$MODE" = "phase" ]; then
    python "$ROOT/examples/train_rlpd.py" "$@" \
        --exp_name=tennis_ball_pick_and_place \
        --encoder_type=vit-grounded \
        --encoder_checkpoint_path="$ROOT/examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded_phase/best.msgpack" \
        --checkpoint_path="$SCRIPT_DIR/2026-08-20b_vit_attnPlusHand_phase_maskobs_SUCCESS" \
        --demo_path="$DEMO_0814" \
        --mask_predictor_checkpoint_path="$MASK_PREDICTOR" \
        --mask_selection_mode=pick_classifier \
        --pick_classifier_checkpoint_path="$PICK_CLF" \
        --learner
else
    echo "unknown MODE='$MODE'. Use gaze3q, replicate or phase." >&2
    exit 1
fi
