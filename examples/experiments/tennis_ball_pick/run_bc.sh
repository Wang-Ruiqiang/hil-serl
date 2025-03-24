export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
python ../../train_bc.py "$@" \
    --exp_name=tennis_ball_pick \
    --bc_checkpoint_path=third_bc_run \
    --demo_path=../../demo_data/tennis_ball_pick_100_demos_2025-03-21_14-08-02.pkl \