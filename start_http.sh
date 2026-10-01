#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
ENV_PREFIX="$($PYTHON -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
args=(--host "${HOST:-0.0.0.0}" --port "${PORT:-40012}")
if [[ -f "${ENV_FILE:-.env}" ]]; then
  args+=(--env-file "${ENV_FILE:-.env}")
elif [[ -n "${ENV_FILE:-}" ]]; then
  echo "找不到 ENV_FILE: $ENV_FILE" >&2
  exit 1
fi
exec "$PYTHON" -m uvicorn app.server:app "${args[@]}" "$@"
