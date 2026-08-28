#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/../hil_env/bin/python}"

# Change these values when visualizing another encoder checkpoint or episode.
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/examples/encoder_training/runs/tennis_ball_pick_and_place_cnn_transformer/best.msgpack}"
DATA_ROOT="${DATA_ROOT:-/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place}"
DEMO_NAME="${DEMO_NAME:-tennis_ball_pick_and_place-2026-08-14_12-18-59}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
MAX_FRAMES="${MAX_FRAMES:-1000}"
FRAME_STRIDE="${FRAME_STRIDE:-20}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/examples/encoder_training/visualizations/pick_place_cnn_transformer_ep0}"

exec "${PYTHON}" "${REPO_ROOT}/examples/encoder_training/visualize_encoder.py" \
  --checkpoint "${CHECKPOINT}" \
  --data_root "${DATA_ROOT}" \
  --demo_name "${DEMO_NAME}" \
  --episode_index "${EPISODE_INDEX}" \
  --max_frames "${MAX_FRAMES}" \
  --frame_stride "${FRAME_STRIDE}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}"
