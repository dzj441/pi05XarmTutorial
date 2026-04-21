#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
source ./enable_uv.sh

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PWD/dataset/ckpt}"

POLICY_DIR="${POLICY_DIR:-$PWD/dataset/ckpt/openpi-assets/checkpoints_pytorch/pi05_libero}"
ASSETS_SRC="${ASSETS_SRC:-$PWD/dataset/ckpt/openpi-assets/checkpoints/pi05_libero/assets}"

if [[ ! -f "$POLICY_DIR/model.safetensors" ]]; then
  echo "[ERROR] Missing PyTorch weights: $POLICY_DIR/model.safetensors"
  exit 1
fi

# Converted PyTorch checkpoint may not include assets.
if [[ ! -e "$POLICY_DIR/assets" ]]; then
  ln -sfn "$ASSETS_SRC" "$POLICY_DIR/assets"
fi

uv run scripts/serve_policy.py \
  --port "${PORT:-8000}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG:-pi05_libero}" \
  --policy.dir="$POLICY_DIR"
