#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"


JAX_CKPT_DIR="${1:-$PWD/dataset/ckpt/openpi-assets/checkpoints/pi05_libero}"
PYTORCH_CKPT_DIR="${2:-$PWD/dataset/ckpt/openpi-assets/checkpoints_pytorch/pi05_libero}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"


echo "CONFIG_NAME:      $CONFIG_NAME"
echo "JAX_CKPT_DIR:     $JAX_CKPT_DIR"
echo "PYTORCH_CKPT_DIR: $PYTORCH_CKPT_DIR"



mkdir -p "$PYTORCH_CKPT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
echo "UV_CACHE_DIR:     $UV_CACHE_DIR"

uv run examples/convert_jax_model_to_pytorch.py \
  --config_name "$CONFIG_NAME" \
  --checkpoint_dir "$JAX_CKPT_DIR" \
  --output_path "$PYTORCH_CKPT_DIR" \


CKPT_DIR=checkpoints/pi05_libero/pi05_libero_jax_8gpu_offline/5000 GPU_ID=0 PORT=8005 bash run_libero_eval_tmux_single_ckpt.sh