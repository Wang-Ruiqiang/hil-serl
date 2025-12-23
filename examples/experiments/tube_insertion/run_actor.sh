export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tube_insertion \
    --checkpoint_path=2025-12_23_0_keyboard_tube_insertion_rl_run \
    --actor \