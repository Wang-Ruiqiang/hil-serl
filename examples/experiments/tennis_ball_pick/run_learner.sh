export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=2025-12-29_0_ball_pick_keyboard_rl_run \
    --demo_path=../../demo_data/tennis_ball_pick_25_demos_2025-12-29_20-02-02.pkl \
    --learner \