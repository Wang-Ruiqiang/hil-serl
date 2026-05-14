ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_rlpd.py "$@" \
    --exp_name=tube_insertion \
    --checkpoint_path=2026-04-02_0_keyboard_tube_rl_run \
    --demo_path=../../demo_data/tube_insertion_25_demos_2026-02-06_02-07-46.pkl \
    --learner \