export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=2026-1-2_0_ball_place_keyboard_rl_run \
    --demo_path=../../demo_data/tennis_ball_place_25_demos_2026-01-02_16-48-32.pkl \
    --learner \