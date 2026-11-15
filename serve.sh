#!/bin/bash
set -euo pipefail

# Serve the rollout viewer and the run artifacts. Re-running this script replaces the process already holding the port
# and its index watcher.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
LOG_DIR="$ROOT_DIR/logs"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-44834}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INDEX_INTERVAL_SECONDS="${ROLLOUT_INDEX_INTERVAL_SECONDS:-2}"
INDEX_PID_FILE="$LOG_DIR/rollouts-index.pid"
SERVER_PID_FILE="$LOG_DIR/serve.pid"

if [ "$#" -ne 0 ]; then
    echo "Usage: ./serve.sh" >&2
    exit 2
fi

kill_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti "tcp:$port" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "Killing existing process on port $port (PID: $pids)"
        kill $pids 2>/dev/null || true
        sleep 1
    fi
}

stop_index_watcher() {
    [ -f "$INDEX_PID_FILE" ] || return 0
    local pid command
    pid="$(sed -n '1p' "$INDEX_PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        if [[ "$command" == *"rollouts_index.py"* ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$INDEX_PID_FILE"
}

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
stop_index_watcher
kill_port "$PORT"

# Build once synchronously, so the first page load never races a missing index.
"$PYTHON_BIN" -u rollouts_index.py

# Start a new session so the server survives an interrupted launching shell.
nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' \
    "$PYTHON_BIN" -u "$ROOT_DIR/viewer/serve.py" \
    --port "$PORT" \
    --bind "$HOST" \
    --root "$ROOT_DIR" \
    > "$LOG_DIR/serve.log" 2>&1 < /dev/null &
server_pid="$!"
printf '%s\n' "$server_pid" > "$SERVER_PID_FILE"

for _ in $(seq 50); do
    if lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
if ! lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
    echo "Server failed to bind $HOST:$PORT — see $LOG_DIR/serve.log" >&2
    exit 1
fi

# The watcher re-reads only step files whose size or mtime changed, and replaces the indexes atomically.
nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' \
    "$PYTHON_BIN" -u "$ROOT_DIR/rollouts_index.py" \
    --watch \
    --interval "$INDEX_INTERVAL_SECONDS" \
    --server-pid "$server_pid" \
    > "$LOG_DIR/rollouts-index.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$INDEX_PID_FILE"

cat <<URLS

GRPO rollouts:
  Training: http://$HOST:$PORT/training.html
  Metrics:  http://$HOST:$PORT/metrics.html
  Files:    http://$HOST:$PORT/output/rollouts/

Open a run, a step's rollouts, or one rollout with:
  http://$HOST:$PORT/training.html?run=<run>&step=<step>&rollout=<index>

Serving $ROOT_DIR in the background (re-run ./serve.sh to restart)
Refreshing rollouts-index.jsonl and runs-index.jsonl every $INDEX_INTERVAL_SECONDS seconds

Copy the Modal runs here first:
  mkdir -p output && uvx modal volume get --force grpo-output rollouts output/ && uvx modal volume get --force grpo-output tb output/

Logs:
  $LOG_DIR/serve.log
  $LOG_DIR/rollouts-index.log
URLS
