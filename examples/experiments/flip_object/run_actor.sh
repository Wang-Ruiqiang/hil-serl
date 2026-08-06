ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
python ../../train_rlpd.py "$@" \
    --exp_name=flip_object \
    --checkpoint_path=2026-08-05_flip_object_rl_ablation_run \
    --enable_tactile=0 \
    --eval_checkpoint_step=35000 \
    --eval_checkpoint_step_interval=0 \
    --eval_max_episode_steps=200 \
    --eval_n_trajs=40 \
    --save_video=True \
    --actor
