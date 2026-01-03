export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=2026-1-2_0_ball_place_keyboard_rl_run \
    --checkpoint_path_pick=2026-1-1_0_ball_pick_keyboard_rl_run \
    --actor \