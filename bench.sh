#!/bin/bash
# Benchmark a running server (./start.sh): decode at 1 and 16 streams, prefill at 1k/4k/16k/32k tokens.
# Writes results/<date>-<gpu>.md
set -euo pipefail

URL="${URL:-http://localhost:8090/v1/chat/completions}"
[ -d .venv ] && . .venv/bin/activate
python3 -c "import tokenizers, huggingface_hub" 2>/dev/null || { echo "ERROR: pip install tokenizers huggingface_hub"; exit 1; }

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')
mkdir -p results; OUT="results/$(date +%Y%m%d)-$GPU.md"
{
  echo "# Nemotron 3.5 Lightning benchmark"; echo; echo "GPU: $GPU"; echo
  echo "## Decode, 1 stream, T=1 (3 reps)"; echo '```'
  python3 bench/decode.py --url "$URL" --streams 1 --repeat 3 --max-tokens 256 --temp 1.0 --show 0
  echo '```'; echo "## Decode, 16 streams, T=1 (2 reps)"; echo '```'
  python3 bench/decode.py --url "$URL" --streams 16 --repeat 2 --max-tokens 256 --temp 1.0 --show 0
  echo '```'; echo "## Prefill (1k/4k/16k/32k)"; echo '```'
  python3 bench/prefill.py --url "$URL"
  echo '```'
} | tee "$OUT"
echo "Saved $OUT"
