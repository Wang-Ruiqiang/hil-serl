export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=2026-1-1_0_ball_pick_keyboard_rl_run \
    --demo_path=../../demo_data/tennis_ball_pick_pick_25_demos_2026-01-01_15-14-22.pkl \
    --learner \