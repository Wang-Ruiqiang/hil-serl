export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path_pick=2025-9-16_0_keyboard_rl_run \
    --checkpoint_path=2025-10-6_0_keyboard_place_rl_run \
    --actor \