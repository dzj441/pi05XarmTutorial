#!/usr/bin/env bash
set -euo pipefail

# OpenPI repo root (this script lives in repo root).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Local LeRobot dataset root (contains libero/).
DATA_ROOT="${DATA_ROOT:-/path/to/your/dataroot}"

# Converted PyTorch base checkpoint (from convert_jax_model_to_pytorch.py).
PYTORCH_CKPT_DIR="${PYTORCH_CKPT_DIR:-$PROJECT_ROOT/dataset/ckpt/openpi-assets/checkpoints_pytorch/pi05_base}"

# Norm-stats assets directory.
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/assets/pi05_libero}"

CONFIG_NAME="pi05_libero"
EXP_NAME="${1:-pi05_libero_pytorch_offline}"
LOG_FILE="$PROJECT_ROOT/train_${EXP_NAME}.log"

# Offline + local paths.
export HF_LEROBOT_HOME="$DATA_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/wandb}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/dataset/ckpt}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Make repo_id=physical-intelligence/libero resolve to local dataset.
mkdir -p "$HF_LEROBOT_HOME/physical-intelligence"
ln -sfn "$HF_LEROBOT_HOME/libero" "$HF_LEROBOT_HOME/physical-intelligence/libero"

# Quick checks.
[[ -f "$PYTORCH_CKPT_DIR/model.safetensors" ]] || {
  echo "[ERROR] Missing PyTorch checkpoint: $PYTORCH_CKPT_DIR/model.safetensors"
  exit 1
}
[[ -f "$ASSETS_DIR/physical-intelligence/libero/norm_stats.json" ]] || {
  echo "[ERROR] Missing norm stats: $ASSETS_DIR/physical-intelligence/libero/norm_stats.json"
  exit 1
}

cd "$PROJECT_ROOT"

echo "CONFIG_NAME:       $CONFIG_NAME"
echo "EXP_NAME:          $EXP_NAME"
echo "DATA_ROOT:         $DATA_ROOT"
echo "PYTORCH_CKPT_DIR:  $PYTORCH_CKPT_DIR"
echo "ASSETS_DIR:        $ASSETS_DIR"
echo "WANDB_MODE:        $WANDB_MODE"
echo "LOG_FILE:          $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE:    $NPROC_PER_NODE"
echo "Batch/worker:      use config defaults unless BATCH_SIZE/NUM_WORKERS is set"

cmd=(
  uv run torchrun --standalone --nnodes=1 --nproc_per_node "$NPROC_PER_NODE" scripts/train_pytorch.py "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --pytorch-weight-path "$PYTORCH_CKPT_DIR"
  --data.assets.assets-dir "$ASSETS_DIR"
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

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"

rc=${PIPESTATUS[0]}
echo "train exit code: $rc"

if [[ $rc -ne 0 ]]; then
  echo "Training failed. Keeping terminal open for debug..."
  exec bash -i
fi
