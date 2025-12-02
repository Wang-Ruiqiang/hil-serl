export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_rlpd.py "$@" \
    --exp_name=twist_bottle_cap \
    --checkpoint_path=2025-11-30_0_keyboard_ablation_bottle_twist_rl_run \
    --demo_path=../../demo_data/twist_bottle_cap_25_demos_2025-11-26_15-01-15.pkl \
    --learner \