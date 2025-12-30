export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tube_insertion \
    --checkpoint_path=2025-12_27_0_keyboard_tube_insertion_ablation_rl_run \
    --demo_path=../../demo_data/tube_insertion_25_demos_2025-12-27_15-55-30.pkl \
    --learner \