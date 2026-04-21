#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="path/to/your/repo"
cd "$REPO_ROOT"

# Usage:
#   bash run_libero_eval_tmux_single_ckpt.sh [CKPT_DIR] [GPU_ID] [PORT] [POLICY_CONFIG]
# Env override also works:
#   CKPT_DIR=... GPU_ID=1 PORT=8001 POLICY_CONFIG=pi05_libero_low_mem_finetune bash run_libero_eval_tmux_single_ckpt.sh
DEFAULT_CKPT_DIR="path/to/your/repo/checkpoints/pi05_libero/pi05_libero_jax_8gpu_offline/29999"
CKPT_DIR="${1:-${CKPT_DIR:-$DEFAULT_CKPT_DIR}}"
GPU_ID="${2:-${GPU_ID:-0}}"
PORT="${3:-${PORT:-8000}}"
POLICY_CONFIG="${4:-${POLICY_CONFIG:-pi05_libero}}"

HOST=127.0.0.1
TASK_SUITE=libero_spatial
NUM_TRIALS=50

RUN_TAG="$(basename "$CKPT_DIR")_gpu${GPU_ID}_p${PORT}"
OUT_DIR="$REPO_ROOT/data/libero_eval_jax/$RUN_TAG"
SESSION_NAME="eval_${RUN_TAG}_$(date +%m%d%H%M%S)"

mkdir -p "$OUT_DIR"

tmux new-session -d -s "$SESSION_NAME" -n server
tmux new-window -t "$SESSION_NAME" -n eval

tmux send-keys -t "$SESSION_NAME:server" "cd $REPO_ROOT" C-m
tmux send-keys -t "$SESSION_NAME:server" \
  "CUDA_VISIBLE_DEVICES=$GPU_ID POLICY_CONFIG=$POLICY_CONFIG POLICY_DIR=$CKPT_DIR PORT=$PORT bash path/to/your/repo/bash_script/eval/run_libero_server.sh" C-m

tmux send-keys -t "$SESSION_NAME:eval" "cd $REPO_ROOT" C-m
tmux send-keys -t "$SESSION_NAME:eval" \
  "sleep 20; HOST=$HOST PORT=$PORT TASK_SUITE=$TASK_SUITE NUM_TRIALS=$NUM_TRIALS VIDEO_OUT_PATH=$OUT_DIR/videos TXT_LOG_PATH=$OUT_DIR/eval_summary.txt CSV_LOG_PATH=$OUT_DIR/eval_episode_results.csv bash path/to/your/repo/bash_script/eval/run_libero_sim_eval_with_log.sh" C-m

echo "Session: $SESSION_NAME"
echo "CKPT: $CKPT_DIR"
echo "POLICY_CONFIG: $POLICY_CONFIG"
echo "GPU_ID: $GPU_ID"
echo "PORT: $PORT"
echo "Logs: $OUT_DIR"
tmux attach -t "$SESSION_NAME"
