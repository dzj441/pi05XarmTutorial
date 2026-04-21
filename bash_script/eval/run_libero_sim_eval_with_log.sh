#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="path/to/your/repo"
cd "$REPO_ROOT"

source examples/libero/.venv/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/third_party/libero"

# Remove leaked torch libs from py3.12 site-packages if present in parent shell.
export LD_LIBRARY_PATH="$(
  echo "${LD_LIBRARY_PATH:-}" \
    | tr ':' '\n' \
    | grep -Ev 'python3\.12/dist-packages/torch|python3\.12/dist-packages/torch_tensorrt' \
    | paste -sd: -
)"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

run_one_suite() {
  local suite="$1"
  local video_out_path="$2"
  local txt_log_path="$3"
  local fail_log_path="$4"
  local csv_log_path="${5:-}"

  local cmd=(
    python examples/libero/eval_with_log.py
    --args.host "${HOST:-127.0.0.1}"
    --args.port "${PORT:-8000}"
    --args.task-suite-name "$suite"
    --args.num-trials-per-task "${NUM_TRIALS:-50}"
    --args.video-out-path "$video_out_path"
    --args.txt-log-path "$txt_log_path"
    --args.fail-log-path "$fail_log_path"
  )
  if [[ -n "$csv_log_path" ]]; then
    cmd+=(--args.csv-log-path "$csv_log_path")
  fi
  "${cmd[@]}"
}

if [[ "${RUN_ALL_SUITES:-0}" == "1" ]]; then
  OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/libero_eval/all_suites}"
  SUMMARY_PATH="${SUMMARY_PATH:-$OUT_ROOT/suite_success_rates.txt}"
  ALL_FAIL_PATH="${ALL_FAIL_PATH:-$OUT_ROOT/all_fail_episodes.txt}"
  SUITES=(${SUITES:-libero_spatial libero_object libero_goal libero_10})

  mkdir -p "$OUT_ROOT"
  echo "suite success_rate" > "$SUMMARY_PATH"
  : > "$ALL_FAIL_PATH"

  for suite in "${SUITES[@]}"; do
    suite_dir="$OUT_ROOT/$suite"
    mkdir -p "$suite_dir/videos"

    run_one_suite \
      "$suite" \
      "$suite_dir/videos" \
      "$suite_dir/eval_summary.txt" \
      "$suite_dir/fail_episodes.txt" | tee "$suite_dir/eval.stdout.log"

    rate="$(grep -E '^Total success rate:' "$suite_dir/eval_summary.txt" | tail -n 1 | sed 's/^Total success rate: //')"
    echo "$suite ${rate:-UNKNOWN}" >> "$SUMMARY_PATH"
    echo "[suite=$suite]" >> "$ALL_FAIL_PATH"
    cat "$suite_dir/fail_episodes.txt" >> "$ALL_FAIL_PATH"
    echo >> "$ALL_FAIL_PATH"
  done

  echo "output root: $OUT_ROOT"
  echo "suite success rates: $SUMMARY_PATH"
  echo "all fail episodes: $ALL_FAIL_PATH"
else
  run_one_suite \
    "${TASK_SUITE:-libero_spatial}" \
    "${VIDEO_OUT_PATH:-$REPO_ROOT/data/libero/videos}" \
    "${TXT_LOG_PATH:-$REPO_ROOT/data/libero/eval_summary.txt}" \
    "${FAIL_LOG_PATH:-$REPO_ROOT/data/libero/fail_episodes.txt}" \
    "${CSV_LOG_PATH:-}"
fi
