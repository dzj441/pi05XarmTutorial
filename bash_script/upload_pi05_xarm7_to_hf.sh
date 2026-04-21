#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-DZJ181u2u/checkpoints}"
LOCAL_DIR="${2:-checkpoints/pi05_xarm7_finetune}"
REPO_TYPE="${HF_REPO_TYPE:-model}"
REVISION="${HF_REVISION:-main}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' command not found."
  echo "Install with: pip install -U \"huggingface_hub[cli]\""
  exit 1
fi

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "Error: local directory not found: ${LOCAL_DIR}"
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "${HF_TOKEN}" >/dev/null
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Error: Hugging Face is not authenticated."
  echo "Run: hf auth login"
  echo "Or set env var: HF_TOKEN=..."
  exit 1
fi

echo "Uploading '${LOCAL_DIR}' to '${REPO_ID}' (repo_type=${REPO_TYPE}, revision=${REVISION})..."
hf upload-large-folder "${REPO_ID}" "${LOCAL_DIR}" \
  --repo-type "${REPO_TYPE}" \
  --revision "${REVISION}"

echo "Upload completed."
