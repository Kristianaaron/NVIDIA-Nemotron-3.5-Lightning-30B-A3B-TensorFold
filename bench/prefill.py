"""Prefill bench through the server: prompts of N tokens (repo source text), TTFT and prefill tok/s."""
import glob, json, os, sys, time, urllib.request, argparse
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--text-dir", default="./tensorfold/zig/src")
ap.add_argument("--url", default="http://localhost:8090/v1/chat/completions")
ap.add_argument("--tokenizer", default="", help="tokenizer.json of the model (default: found via huggingface_hub)")
a = ap.parse_args()

if a.tokenizer:
    tok_path = a.tokenizer
else:
    from huggingface_hub import hf_hub_download
    tok_path = os.path.join(os.environ.get("WEIGHTS_DIR", "./weights"), "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = hf_hub_download(os.environ.get("WEIGHTS_REPO", "Krishando/Nemotron-3.5-Lightning-30B-A3B-EXL3-2.3bpw"), "tokenizer.json")
tok = Tokenizer.from_file(tok_path)
files = sorted(glob.glob(os.path.join(a.text_dir, "**/*.zig"), recursive=True))
corpus = "".join(open(f).read() for f in files)
ids = tok.encode(corpus).ids
lengths = [1024, 4096, 16384, 32000]
off = int(os.environ.get('PREFILL_OFFSET', '0'))
for n in lengths:
    text = tok.decode(ids[off:off + n]); off += n
    body = {"model": "nemo", "messages": [{"role": "user", "content": "Summarise what this code does in one sentence.\n\n" + text}],
            "max_tokens": 16, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; out = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
            ev = json.loads(line[5:])
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                dl = ch.get("delta", {}); d = dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or ""
                if d and first is None: first = time.time()
                out.append(d)
    pt = (usage or {}).get("prompt_tokens")
    ttft = (first or time.time()) - t0
    print(f"PREFILL {n} target, {pt} prompt tokens: TTFT {ttft:.2f}s -> {pt / ttft:.0f} tok/s; reply {''.join(out)[:80]!r}", flush=True)
