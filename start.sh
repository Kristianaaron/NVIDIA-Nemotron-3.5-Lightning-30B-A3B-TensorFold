#!/bin/bash
# 16 streams x 64k context, one shared 128k-token KV pool, MTP drafts (1 per stream), OpenAI-compatible API.
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"
PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"
BIN=./tensorfold/zig-out/native/bin/tensorfold-native

[ -x "$BIN" ] || { echo "ERROR: engine not built; run ./install.sh"; exit 1; }
[ -f "$WEIGHTS_DIR/config.json" ] || { echo "ERROR: weights not found in $WEIGHTS_DIR; run ./install.sh"; exit 1; }

# One stream keeps a full 64k window; the other 15 share 65,024 more tokens of KV pages (128k in all).
export TF_NEMOTRON_KV_POOL_TOKENS=65024
# One CUDA graph per decode window, shared across streams (output unchanged).
export TF_NEMOTRON_SHARED_GRAPHS=1

exec "$BIN" serve "$WEIGHTS_DIR" --name nemo --host "$HOST" --port "$PORT" \
  --parallel 16 --context 65536 --mtp-drafts 1
