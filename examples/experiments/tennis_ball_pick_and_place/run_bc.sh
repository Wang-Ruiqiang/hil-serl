export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 && \
python ../../train_bc.py "$@" \
    --exp_name=tennis_ball_pick_and_place \
    --bc_checkpoint_path=2025-4-23_0_bc_run \
    --demo_path=../../demo_data/tennis_ball_pick_and_place_100_demos_2025-03-21_14-08-02.pkl \
