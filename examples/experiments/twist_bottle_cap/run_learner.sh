export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_rlpd.py "$@" \
    --exp_name=twist_bottle_cap \
    --checkpoint_path=2025-11-18_0_keyboard_bottle_twist_rl_run \
    --demo_path=../../demo_data/twist_bottle_cap_22_demos_2025-11-17_21-53-01.pkl \
    --learner \