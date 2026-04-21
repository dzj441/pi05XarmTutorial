#!/usr/bin/env bash
set -euo pipefail

# OpenPI repo root (fixed absolute path).
PROJECT_ROOT="path/to/your/repo"

CONFIG_NAME="pi05_xarm7_finetune"
EXP_NAME="${1:-pi05_libero_jax_8gpu_realworld}"
LOG_FILE="$PROJECT_ROOT/train_${EXP_NAME}.log"

# Local LeRobot dataset root (fixed absolute path).
DATA_ROOT="path/to/your/repo/dataset/lerobot"

# Use locally computed norm stats: $ASSETS_BASE_DIR/pi05_libero/physical-intelligence/libero/norm_stats.json
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$PROJECT_ROOT/assets}"
NORM_STATS_PATH="$ASSETS_BASE_DIR/pi05_xarm7_finetune/local/xarm7_data1/norm_stats.json"

# Local JAX base checkpoint params for initialization.
JAX_BASE_PARAMS="${JAX_BASE_PARAMS:-$PROJECT_ROOT/dataset/ckpt/openpi-assets/checkpoints/pi05_base/params}"

# Offline + local paths.
export HF_LEROBOT_HOME="$DATA_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/wandb}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/dataset/ckpt}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# 8-GPU setup (single node).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"


cd "$PROJECT_ROOT"

echo "CONFIG_NAME:       $CONFIG_NAME"
echo "EXP_NAME:          $EXP_NAME"
echo "DATA_ROOT:         $DATA_ROOT"
echo "ASSETS_BASE_DIR:   $ASSETS_BASE_DIR"
echo "NORM_STATS_PATH:   $NORM_STATS_PATH"
echo "JAX_BASE_PARAMS:   $JAX_BASE_PARAMS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "WANDB_MODE:        $WANDB_MODE"
echo "LOG_FILE:          $LOG_FILE"

cmd=(
  uv run scripts/train.py "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --assets-base-dir "$ASSETS_BASE_DIR"
  --weight-loader.params-path "$JAX_BASE_PARAMS"
)

# Optional overrides. If not set, official config values are used.
if [[ -n "${BATCH_SIZE:-}" ]]; then
  cmd+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  cmd+=(--num-workers "$NUM_WORKERS")
fi
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
  cmd+=(--save-interval "$SAVE_INTERVAL")
fi
if [[ -n "${NUM_TRAIN_STEPS:-}" ]]; then
  cmd+=(--num-train-steps "$NUM_TRAIN_STEPS")
fi


if [[ -n "${OVERWRITE:-}" ]]; then
  if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
    cmd+=(--overwrite)
  fi
fi
if [[ -n "${RESUME:-}" ]]; then
  if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi
fi
if [[ -n "${FSDP_DEVICES:-}" ]]; then
  cmd+=(--fsdp-devices "$FSDP_DEVICES")
fi

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
echo "train exit code: $rc"

if [[ $rc -ne 0 ]]; then
  echo "Training failed. Keeping terminal open for debug..."
  exec bash -i
fi
