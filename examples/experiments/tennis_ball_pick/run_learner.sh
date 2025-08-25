export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=2025-8-21_keyboard_rl_run \
    --demo_path=../../demo_data/tennis_ball_pick_100_demos_2025-08-21_17-16-43.pkl \
    --learner \