#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="path/to/your/repo"
cd "$REPO_ROOT"

source .venv/bin/activate
source ./enable_uv.sh

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/dataset/ckpt}"

uv run scripts/serve_policy.py \
  --port "${PORT:-8000}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG:-pi05_libero}" \
  --policy.dir="${POLICY_DIR:-$REPO_ROOT/dataset/ckpt/openpi-assets/checkpoints/pi05_libero}"
