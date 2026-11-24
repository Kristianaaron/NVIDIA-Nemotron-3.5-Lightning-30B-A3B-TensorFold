"""Streamed single/multi-stream bench against an OpenAI server: TTFT, decode tok/s (after first token), aggregate."""
import json, sys, time, threading, urllib.request, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8090/v1/chat/completions")
ap.add_argument("--streams", type=int, nargs="+", default=[1])
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--prompt", default="Write a Python function that reverses a singly linked list, with a short docstring and an example.")
ap.add_argument("--show", type=int, default=400)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--no-thinking", action="store_true")
a = ap.parse_args()

PROMPTS = [a.prompt, "Explain in five sentences how a hash map handles collisions.",
           "Write a bash one-liner that counts lines in all .py files recursively, then explain it.",
           "Write a Python class for an LRU cache with get and put in O(1).",
           "What is the difference between a process and a thread? Answer briefly.",
           "Write a SQL query that returns the top 3 salaries per department.",
           "Write a JavaScript debounce function and explain when to use it.",
           "Summarise the rules of binary search and write it in Python."]

OFF = [0]
def one(i, res):
    body = {"model": "nemo", "messages": [{"role": "user", "content": PROMPTS[(i + OFF[0]) % len(PROMPTS)]}],
            "max_tokens": a.max_tokens, "temperature": a.temp, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": not a.no_thinking}}
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = []; usage = None; times = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            ev = json.loads(data)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                dl = ch.get("delta", {}); d = dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or ""
                if d:
                    now = time.time(); times.append(now)
                    if first is None: first = now
                    n += 1; text.append(d)
    end = time.time()
    toks = (usage or {}).get("completion_tokens") or n
    res[i] = dict(ttft=(first or end) - t0, total=end - t0, toks=toks, chunks=n,
                  decode=(toks - 1) / (end - first) if first and end > first else 0.0, text="".join(text),
                  prompt_tokens=(usage or {}).get("prompt_tokens"))

ALL = []
for k in a.streams:
    for rep in range(a.repeat):
        OFF[0] = rep
        res = [None] * k
        th = [threading.Thread(target=one, args=(i, res)) for i in range(k)]
        t0 = time.time()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.time() - t0
        tot = sum(r["toks"] for r in res)
        print(f"streams {k} rep {rep}: wall {wall:.2f}s total tokens {tot} aggregate {tot / wall:.1f} tok/s; "
              f"per-stream decode {[round(r['decode'], 1) for r in res]} ttft {[round(r['ttft'], 2) for r in res]} "
              f"prompt_tokens {[r['prompt_tokens'] for r in res]}", flush=True)
        ALL.extend(res)
        if a.show and rep == 0:
            print("  text[0]:", repr(res[0]["text"][:a.show]), flush=True)

if len(a.streams) == 1 and a.streams[0] == 1 and a.repeat > 1:
    d = [r["decode"] for r in ALL[1:]]
    print(f"SUMMARY single-stream decode mean (excluding first) {sum(d) / len(d):.1f} tok/s; min {min(d):.1f} max {max(d):.1f}", flush=True)
