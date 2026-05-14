ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_bc.py "$@" \
    --exp_name=tennis_ball_pick \
    --bc_checkpoint_path=2026-4-6_0_bc_run \
    --demo_path=../../demo_data/tennis_ball_pick_20_demos_2026-01-14_15-56-09.pkl \
