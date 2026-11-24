# NVIDIA Nemotron 3.5 Lightning 30B-A3B: 16 streams on one 16 GB GPU with TensorFold

NVIDIA's Nemotron 3.5 Lightning (30B total, ~3B active, hybrid Mamba-attention MoE) quantized to fit fully in the
VRAM of one RTX 5070 Ti (16 GB), serving **16 concurrent streams at up to 64k context each** on the TensorFold
native engine, with the model's own MTP head as a drafter.

- Weights: [Hugging Face](https://huggingface.co/Krishando/Nemotron-3.5-Lightning-30B-A3B-EXL3-2.3bpw)
- Engine: TensorFold fork, branch [`nemotron-lightning-5070ti`](https://github.com/Kristianaaron/TensorFold/tree/nemotron-lightning-5070ti)

## Results

Measured on an RTX 5070 Ti (16 GB), WSL2, 2026-11-23/24. Thinking on.

### Speed

| Test | What it measures | Result |
|---|---|---|
| 1 stream, T=1 | decode speed of a single chat (256-token replies, short prompt) | **392 tok/s** |
| 1 stream, greedy | same, temperature 0 | 376 tok/s |
| 16 streams, T=1, total | decode speed with 16 chats at once, steady state | **1,040 tok/s** (~65 tok/s per stream) |
| 16 streams, greedy, total | same, temperature 0 | 1,013 tok/s |
| 16-agent swarm, total | 16 tool-using agents for 10 minutes (4k system prompt, 600-token tasks, 400-token replies, 300-token tool results) | **443 tok/s**, 681 turns, 0 failed |
| Swarm time to first token | median / 95th percentile | 11.1 s / 14.2 s |
| Prefill, 1 stream | prompt processing; time to first token | 1k: 1,770 tok/s (0.6 s) · 4k: 3,830 tok/s (1.1 s) · 16k: 4,010 tok/s (4.1 s) · 32k: 4,055 tok/s (7.9 s) |

Drafting with the MTP head does not change output: greedy and seeded T=1 outputs are bit-identical with drafts on and off.

### Quality vs the BF16 original

Every comparison is against NVIDIA's BF16 release, with the same prompts, settings and token budgets.

| Test | What it measures | This quant | BF16 original |
|---|---|---|---|
| Research questions | correct answers out of 600 | **301** | 301 |
| Research questions | loops and malformed output | no worse than BF16 | reference |
| Tool calls | valid, well-formed tool calls (550) | no worse than BF16 | reference |
| CUDA/Triton kernel writing (8k token cap) | replies that fall into repetition | **4.2%** (50 of 1,200) | 2.0% (12 of 600) |
| All tests | malformed output | 0 | 0 |

Known gap: on long CUDA/Triton kernel-writing replies that think up to the 8k token cap, this quant falls into
repetition about twice as often as the BF16 original (4.2% vs 2.0%). Everything else measured matches the original.

### Size

| Part | Size |
|---|---|
| Routed experts (5,980 matrices) | 8.35 GiB, EXL3 at 2 to 3.5 bits per matrix, 2.31 bits average |
| MTP head experts (260 matrices) | EXL3, 3.5 bits |
| Rest of the model (attention, Mamba, shared experts, router, embeddings, head, MTP head) | 4-bit affine, groups of 64 |
| Whole model folder | 11.6 GB |

## Quick start

Requirements: an NVIDIA GPU with 16 GB of VRAM (tested: RTX 5070 Ti), CUDA 13, Linux or WSL2, Zig 0.17.0.

```bash
./install.sh     # builds the engine and downloads the weights
./start.sh       # 16 streams x 64k context, OpenAI-compatible API on localhost:8090
```

`start.sh` runs:

```bash
TF_NEMOTRON_KV_POOL_TOKENS=65024 TF_NEMOTRON_SHARED_GRAPHS=1 \
  ./tensorfold/zig-out/native/bin/tensorfold-native serve ./weights --name nemo --port 8090 \
  --parallel 16 --context 65536 --mtp-drafts 1
```

## Presets

| Preset | Streams | Context per stream | Shared KV pool | Use |
|---|---|---|---|---|
| default | 16 | 64k | 128k tokens | agent swarms, many chats |

Streams share one KV pool: 16 long chats at once will queue for pages; short chats run fully in parallel.

## Benchmark it yourself

```bash
./bench.sh
```

`bench.sh` times each request from the client, start-up included, so its numbers sit below the steady-state
figures above. On an RTX 5070 Ti expect about 340 tok/s per request at 1 stream and about 700 tok/s total at
16 streams, with prefill as in the table.

## Known issues

- Long prompts stall other streams while they are processed (no chunked prefill yet); this is the main reason the
  swarm total is below the clean 16-stream number.
- No VRAM is left for other GPU work while the server runs.

## Credits

See [CREDITS.md](CREDITS.md).

## License

This recipe repo is MIT-licensed (see [LICENSE](LICENSE)). The quantized weights follow NVIDIA's
[OpenMDW-1.1](https://openmdw.ai/license/1-1/) license of the original model. The engine is Apache-2.0 (see
[CREDITS.md](CREDITS.md)).
