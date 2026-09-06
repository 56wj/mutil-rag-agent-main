#!/usr/bin/env bash
# 启动 FastAPI (多 uvicorn worker)。改造文档第 10 步。
# 用法: scripts/run_api.sh
#   UVICORN_WORKERS=4 APP_PORT=9900 PYTHON=python scripts/run_api.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
WORKERS="${UVICORN_WORKERS:-4}"
PORT="${APP_PORT:-9900}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# prometheus_client 多进程模式要求父进程启动前清理共享目录。
# 目录放 .run 下，避免跨项目/跨启动残留 Gauge 文件。
if [[ "${METRICS_ENABLED:-true}" == "true" ]]; then
  export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-$ROOT/.run/prometheus_multiproc}"
  mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
  rm -f "$PROMETHEUS_MULTIPROC_DIR"/*.db
fi

echo "[run_api] uvicorn workers=$WORKERS port=$PORT (python=$PYTHON)"
echo "[run_api] 提示: 多 uvicorn worker 下并发上限由 Redis 分布式槽全局保证, 不会被进程数放大"
exec "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
