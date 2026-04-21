#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run this script with: source ./enable_uv.sh" >&2
  exit 1
fi

UV_ENV_FILE="path/to/uv/env/file"

if [[ ! -f "${UV_ENV_FILE}" ]]; then
  echo "uv env file not found: ${UV_ENV_FILE}" >&2
  return 1
fi

# shellcheck disable=SC1090
if ! source "${UV_ENV_FILE}"; then
  echo "failed to source uv env: ${UV_ENV_FILE}" >&2
  return 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH after source: ${UV_ENV_FILE}" >&2
  return 1
fi

echo "uv enabled: $(command -v uv)"
uv --version || echo "warning: unable to run 'uv --version'" >&2

