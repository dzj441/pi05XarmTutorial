#!/usr/bin/env bash
set -euo pipefail

# Fixed paths/config (minimal script).
PROJECT_ROOT="path/to/your/repo"
CONFIG_NAME="pi05_libero_low_mem_finetune"
EXP_NAME="${1:-pi05_libero_lora_jax_offline}"
LOG_FILE="$PROJECT_ROOT/train_${EXP_NAME}.log"

DATA_ROOT="/path/to/your/dataroot"
ASSETS_BASE_DIR="$PROJECT_ROOT/assets"
# Reuse the existing pi05_libero norm stats for LoRA training.
NORM_STATS_PATH="$ASSETS_BASE_DIR/pi05_libero/physical-intelligence/libero/norm_stats.json"
ASSETS_DIR_FOR_DATA="$ASSETS_BASE_DIR/pi05_libero"
JAX_BASE_PARAMS="$PROJECT_ROOT/dataset/ckpt/openpi-assets/checkpoints/pi05_base/params"

export HF_LEROBOT_HOME="$DATA_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export OPENPI_DATA_HOME="$PROJECT_ROOT/dataset/ckpt"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

[[ -f "$NORM_STATS_PATH" ]] || { echo "[ERROR] Missing norm stats: $NORM_STATS_PATH"; exit 1; }
[[ -f "$JAX_BASE_PARAMS/_METADATA" ]] || { echo "[ERROR] Missing JAX base params: $JAX_BASE_PARAMS"; exit 1; }

cd "$PROJECT_ROOT"

uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --data.assets.assets-dir "$ASSETS_DIR_FOR_DATA" \
  --weight-loader.params-path "$JAX_BASE_PARAMS" \
  2>&1 | tee "$LOG_FILE"

rc=${PIPESTATUS[0]}
echo "train exit code: $rc"

if [[ $rc -ne 0 ]]; then
  echo "Training failed. Keeping terminal open for debug..."
  exec bash -i
fi
