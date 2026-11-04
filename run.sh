#!/usr/bin/env bash
# Odd One Out: one-command launcher for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/sha256rma/agent-detector/main/run.sh | bash
#
# What it does, in order:
#   1. Installs uv (a Python package manager) if you don't have it.
#   2. Downloads this project into ~/agent-detector, or updates it if it's already there.
#   3. Installs Python 3.12 and the dependencies into that folder only (nothing system-wide).
#   4. Starts the server on http://127.0.0.1:8000 and opens it in your browser.
# The first run downloads about 500 MB (Python packages plus the 270 MB model) and takes a few
# minutes. Later runs start in seconds. Press Ctrl+C to stop the server.
set -euo pipefail

REPO="https://github.com/sha256rma/agent-detector.git"
DIR="${AGENT_DETECTOR_DIR:-$HOME/agent-detector}"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# Use the current folder if this script is being run from inside a checkout.
if [ -f "pyproject.toml" ] && [ -f "app/main.py" ]; then
  DIR="$(pwd)"
else
  command -v git >/dev/null || { echo "Please install git first, then run this again."; exit 1; }
  if [ -d "$DIR/.git" ]; then
    say "Updating $DIR"
    git -C "$DIR" pull --ff-only
  else
    say "Downloading the project into $DIR"
    git clone --depth 1 "$REPO" "$DIR"
  fi
fi
cd "$DIR"

if ! command -v uv >/dev/null; then
  say "Installing uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

say "Installing dependencies (first run: a few minutes)"
uv sync --no-dev

if curl -fs "$URL/api/status" >/dev/null 2>&1; then
  echo "Something is already running on port $PORT. Stop it, or run with another port: PORT=8001 bash run.sh"
  exit 1
fi

say "Starting the server at $URL"
echo "The first start also downloads the model (about 270 MB)."
SELF_BASE="$URL" uv run --no-dev uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM

# Wait until the page is up, then open it.
for _ in $(seq 1 120); do
  curl -fs "$URL/api/status" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "The server stopped unexpectedly. See the messages above."; exit 1; }
  sleep 1
done
say "Ready: $URL  (press Ctrl+C to stop)"
if command -v open >/dev/null; then open "$URL"; elif command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1 || true; fi
wait $SERVER
