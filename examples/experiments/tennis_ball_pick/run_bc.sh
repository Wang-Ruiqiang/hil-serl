export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 && \
python ../../train_bc.py "$@" \
    --exp_name=tennis_ball_pick \
    --bc_checkpoint_path=2025-4-8_bc_run \
    --demo_path=../../demo_data/tennis_ball_pick_100_demos_2025-04-07_15-23-22.pkl \