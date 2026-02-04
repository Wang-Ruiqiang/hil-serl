ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
python ../../train_rlpd.py "$@" \
    --exp_name=lid_grip \
    --checkpoint_path=2026-02-02_keyboard_lid_grip_ablation_rl_run \
    --demo_path=../../demo_data/lid_grip_ablation_25_demos_2026-02-03_15-23-20.pkl \
    --learner