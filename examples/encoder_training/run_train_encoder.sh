# Offline pretraining for the vit-grounded encoder.
# Uses ONLY the 40 demos recorded at t=0 -- no RL replay data.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# "$@" goes LAST so caller overrides win -- argparse keeps the final value of
# a repeated flag, so putting it first silently ignores e.g. --epochs.
python "$SCRIPT_DIR/train_encoder.py" \
    --exp_name=tennis_ball_pick_and_place \
    --frame_stride=1 \
    --epochs=60 \
    --batch_size=64 \
    "$@"
