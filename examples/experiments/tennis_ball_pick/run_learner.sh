export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tennis_ball_pick \
    --checkpoint_path=first_run \
    --demo_path=../../demo_data/tennis_ball_pick_100_demos_2025-02-18_16-58-11.pkl \
    --learner \