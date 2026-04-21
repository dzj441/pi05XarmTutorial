#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="path/to/your/repo"
cd "$REPO_ROOT"

CKPT_DIR="${CKPT_DIR:-path/to/your/repo/dataset/ckpt/openpi-assets/checkpoints/pi05_libero}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SERVER_WAIT_SEC="${SERVER_WAIT_SEC:-20}"
SESSION_NAME="eval_pi05_libero_4suites_g${GPU_ID}_p${PORT}_$(date +%m%d%H%M%S)"
OUT_ROOT="$REPO_ROOT/data/libero_eval_pi05_libero_4suites/$(date +%Y%m%d_%H%M%S)_g${GPU_ID}_p${PORT}"

mkdir -p "$OUT_ROOT"

tmux new-session -d -s "$SESSION_NAME" -n server
tmux new-window -t "$SESSION_NAME" -n eval

tmux send-keys -t "$SESSION_NAME:server" "cd $REPO_ROOT" C-m
tmux send-keys -t "$SESSION_NAME:server" \
  "CUDA_VISIBLE_DEVICES=$GPU_ID POLICY_CONFIG=$POLICY_CONFIG POLICY_DIR=$CKPT_DIR PORT=$PORT bash $REPO_ROOT/bash_script/eval/run_libero_server.sh | tee $OUT_ROOT/server.log" C-m

tmux send-keys -t "$SESSION_NAME:eval" "cd $REPO_ROOT" C-m
tmux send-keys -t "$SESSION_NAME:eval" \
  "sleep $SERVER_WAIT_SEC; HOST=127.0.0.1 PORT=$PORT NUM_TRIALS=$NUM_TRIALS OUT_ROOT=$OUT_ROOT RUN_ALL_SUITES=1 bash $REPO_ROOT/bash_script/eval/run_libero_sim_eval_with_log.sh" C-m

echo "Session: $SESSION_NAME"
echo "CKPT: $CKPT_DIR"
echo "POLICY_CONFIG: $POLICY_CONFIG"
echo "GPU_ID: $GPU_ID"
echo "PORT: $PORT"
echo "Output Root: $OUT_ROOT"
tmux attach -t "$SESSION_NAME"
