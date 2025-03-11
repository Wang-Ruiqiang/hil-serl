export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=second_run \
    --demo_path=../../demo_data/tennis_ball_pick_100_demos_2025-03-10_18-07-48.pkl \
    --learner \