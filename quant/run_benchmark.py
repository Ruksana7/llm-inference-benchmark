"""Evaluate quantized variants on WikiText-2 perplexity and GSM8K accuracy.

Usage:
  python -m quant.run_benchmark --model Qwen2.5-1.5B-Instruct --gsm8k-n 200 --out results/quant_1.5b.jsonl
"""
import argparse
import json
import math
import time

import torch
from datasets import load_dataset

from common import chat_ids, device, extract_number, gsm8k_items, load_model, sync
from quant.quantize import apply_awq, apply_rtn, collect_activations, packed_bytes

# (method, bits, group_size). group_size -1 = one group per output row (per-channel)
CONFIGS = [
    ("fp16", 16, 0),
    ("rtn", 8, -1),
    ("rtn", 4, -1),
    ("rtn", 4, 128),
    ("rtn", 4, 32),
    ("awq", 4, 128),
    ("rtn", 3, 128),
    ("awq", 3, 128),
]


@torch.no_grad()
def wikitext_ppl(model, tok, ctx=1024, max_windows=None):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    nll, count = 0.0, 0
    windows = range(0, ids.numel() - 1, ctx)
    for wi, start in enumerate(windows):
        if max_windows and wi >= max_windows:
            break
        chunk = ids[start : start + ctx + 1].unsqueeze(0).to(model.device)
        if chunk.shape[1] < 2:
            break
        logits = model(chunk[:, :-1]).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0], chunk[0, 1:], reduction="sum")
        nll += loss.item()
        count += chunk.shape[1] - 1
    return math.exp(nll / count)


@torch.no_grad()
def gsm8k_accuracy(model, tok, items, max_new=320, batch=16):
    tok.padding_side = "left"
    correct = 0
    outputs = []
    for b in range(0, len(items), batch):
        chunk = items[b : b + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
                 for q, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for (q, gold), g in zip(chunk, gen):
            ans = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_number(ans)
            ok = pred is not None and float(pred) == float(gold)
            correct += ok
            outputs.append({"gold": gold, "pred": pred, "ok": bool(ok)})
    return correct / len(items), outputs


def calib_batches(tok, n=64, length=512):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:20000])
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * length : (i + 1) * length].unsqueeze(0) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--configs", type=int, nargs="*", help="indices into CONFIGS (default: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok, _ = load_model(args.model)
    items = gsm8k_items(args.gsm8k_n)
    acts = None
    chosen = [CONFIGS[i] for i in args.configs] if args.configs else CONFIGS

    with open(args.out, "a") as f:
        for method, bits, gs in chosen:
            _, model = load_model(args.model)  # fresh fp16 weights each time
            t0 = time.perf_counter()
            extra = {}
            if method == "rtn":
                apply_rtn(model, bits, gs)
            elif method == "awq":
                if acts is None:
                    acts = collect_activations(model, calib_batches(tok))
                alphas = apply_awq(model, bits, gs, acts)
                extra["mean_alpha"] = sum(alphas.values()) / len(alphas)
            sync()
            qtime = time.perf_counter() - t0
            ppl = wikitext_ppl(model, tok)
            acc, outs = gsm8k_accuracy(model, tok, items)
            rec = {"model": args.model, "method": method, "bits": bits, "group_size": gs,
                   "size_gb": packed_bytes(model, bits, gs) / 1e9, "wikitext2_ppl": ppl,
                   "gsm8k_acc": acc, "gsm8k_n": len(items), "quant_time_s": qtime, **extra}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            with open(args.out.replace(".jsonl", f"_{method}{bits}g{gs}_gsm8k.json"), "w") as g:
                json.dump(outs, g)
            print(json.dumps(rec), flush=True)
            del model
            if device() == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
