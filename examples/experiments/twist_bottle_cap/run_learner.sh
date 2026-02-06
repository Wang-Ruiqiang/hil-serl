ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
python ../../train_rlpd.py "$@" \
    --exp_name=twist_bottle_cap \
    --checkpoint_path=2026-02-04_keyboard_twist_bottle_cap_ablation_rl_run \
    --demo_path=../../demo_data/twist_bottle_cap_ablation_25_demos_2026-02-04_18-55-52.pkl \
    --learner