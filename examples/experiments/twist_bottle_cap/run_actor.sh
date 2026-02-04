ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
python ../../train_rlpd.py "$@" \
    --exp_name=lid_grip \
    --checkpoint_path=2026-02-02_keyboard_lid_grip_ablation_rl_run \
    --checkpoint_path_pick=2026-01-31_keyboard_lid_grip_rl_run \
    --actor