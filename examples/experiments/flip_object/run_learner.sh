ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
python ../../train_rlpd.py "$@" \
    --exp_name=flip_object \
    --checkpoint_path=2026-08-05_flip_object_rl_ablation_run \
    --enable_tactile=0 \
    --demo_path="$ROOT/examples/demo_data/flip_object_ablation*.pkl" \
    --learner