#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source examples/libero/.venv/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"

# Remove leaked torch libs from py3.12 site-packages if present in parent shell.
export LD_LIBRARY_PATH="$(
  echo "${LD_LIBRARY_PATH:-}" \
    | tr ':' '\n' \
    | grep -Ev 'python3\.12/dist-packages/torch|python3\.12/dist-packages/torch_tensorrt' \
    | paste -sd: -
)"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

OUT_DIR="${OUT_DIR:-data/libero_eval_pytorch}"
mkdir -p "$OUT_DIR"

python examples/libero/eval_with_log.py \
  --args.host "${HOST:-127.0.0.1}" \
  --args.port "${PORT:-8000}" \
  --args.task-suite-name "${TASK_SUITE:-libero_spatial}" \
  --args.num-trials-per-task "${NUM_TRIALS:-50}" \
  --args.txt-log-path "${TXT_LOG_PATH:-$OUT_DIR/eval_summary.txt}" \
  --args.csv-log-path "${CSV_LOG_PATH:-$OUT_DIR/eval_episode_results.csv}"
