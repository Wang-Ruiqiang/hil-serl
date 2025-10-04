export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=2025-10-4_0_keyboard_place_rl_run \
    --demo_path=../../demo_data/tennis_ball_pick_place_100_demos_2025-09-27_18-39-25.pkl \
    --learner \