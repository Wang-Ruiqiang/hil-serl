ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
python ../../train_rlpd.py "$@" \
    --exp_name=twist_bottle_cap \
    --checkpoint_path=2026-01-07_keyboard_ablation_bottle_twist_rl_run \
    --actor