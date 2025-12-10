export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tube_insertion \
    --checkpoint_path=2025-12_09_0_keyboard_tube_insertion_rl_run \
    --demo_path=../../demo_data/tube_insertion_25_demos_2025-12-09_21-49-37.pkl \
    --learner \