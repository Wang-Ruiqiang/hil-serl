export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tube_insertion \
    --checkpoint_path=2025-12_03_0_keyboard_ablation_bottle_twist_rl_run \
    --demo_path=../../demo_data/twist_bottle_cap_25_demos_2025-12-03_19-12-08.pkl \
    --learner \