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

python examples/libero/main.py \
  --args.host "${HOST:-127.0.0.1}" \
  --args.port "${PORT:-8000}" \
  --args.task-suite-name "${TASK_SUITE:-libero_spatial}" \
  --args.num-trials-per-task "${NUM_TRIALS:-2}"
