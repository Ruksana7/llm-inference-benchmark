"""Benchmark speculative decoding vs plain greedy decoding.

Usage:
  python -m specdec.run_benchmark --target Qwen2.5-3B-Instruct --draft Qwen2.5-0.5B-Instruct \
      --ks 1 2 3 4 6 8 --n 20 --max-new 128 --out results/specdec_3b.jsonl
"""
import argparse
import json

import torch

from common import chat_ids, device, load_model, prompts
from specdec.speculative import greedy_generate, speculative_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--draft", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--tasks", nargs="+", default=["math", "chat", "code"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok, target = load_model(args.target)
    _, draft = load_model(args.draft)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    dev = device()

    # warm-up so first-call kernel compilation doesn't pollute timings
    w = chat_ids(tok, "Hello").to(dev)
    greedy_generate(target, w, 8, eos)
    speculative_generate(target, draft, w, 8, 2, eos)

    with open(args.out, "w") as f:
        for task in args.tasks:
            for i, p in enumerate(prompts(task, args.n)):
                ids = chat_ids(tok, p).to(dev)
                base, bstats = greedy_generate(target, ids, args.max_new, eos)
                rec = {"task": task, "idx": i, "method": "greedy", "k": 0, "tokens": len(base), **bstats}
                f.write(json.dumps(rec) + "\n")
                for k in args.ks:
                    gen, s = speculative_generate(target, draft, ids, args.max_new, k, eos)
                    # first position where outputs differ (fp16 noise can flip near-ties)
                    m = min(len(gen), len(base))
                    div = next((j for j in range(m) if gen[j] != base[j]), None)
                    if div is None and len(gen) != len(base):
                        div = m
                    rec = {"task": task, "idx": i, "method": "spec", "k": k, "tokens": len(gen),
                           "identical": div is None, "first_divergence": div, **s}
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} {i + 1}/{args.n} done", flush=True)
                if dev == "mps":
                    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
