"""Greedy speculative decoding, implemented from scratch with KV caches.

A small draft model proposes k tokens autoregressively; the large target model
scores all k in ONE forward pass and keeps the longest prefix that matches its
own greedy choice, plus one "bonus" token from its own prediction. With greedy
decoding the output should match plain target decoding token-for-token
(up to fp16 numerical noise from batched vs single-token attention).

Cache invariants (committed = prompt + every accepted token):
  target cache holds committed[:-1]   (the last committed token is fed next round)
  draft  cache holds committed[:dlen]
"""
import time

import torch

from common import sync


def _crop_to(cache, length):
    """Truncate a KV cache to `length` tokens (transformers 5 takes a negative 'remove n')."""
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


@torch.no_grad()
def greedy_generate(model, input_ids, max_new, eos_ids):
    """Plain autoregressive greedy decoding with KV cache (the baseline)."""
    sync()
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    cache = out.past_key_values
    tok = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = [tok.item()]
    while len(gen) < max_new and gen[-1] not in eos_ids:
        out = model(tok, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(tok.item())
    sync()
    return gen, {"time_s": time.perf_counter() - t0, "target_forwards": len(gen)}


@torch.no_grad()
def speculative_generate(target, draft, input_ids, max_new, k, eos_ids):
    dev = input_ids.device
    sync()
    t0 = time.perf_counter()

    t_out = target(input_ids, use_cache=True)
    t_cache = t_out.past_key_values
    first = t_out.logits[:, -1].argmax(-1).item()
    d_cache = draft(input_ids, use_cache=True).past_key_values

    committed = input_ids[0].tolist() + [first]
    prompt_len = input_ids.shape[1]
    dlen = prompt_len
    drafted = accepted = steps = 0
    accept_hist = [0] * (k + 1)

    while len(committed) - prompt_len < max_new and committed[-1] not in eos_ids:
        # ---- draft k tokens ----
        feed = torch.tensor([committed[dlen:]], device=dev)
        drafts = []
        for _ in range(k):
            d_out = draft(feed, past_key_values=d_cache, use_cache=True)
            d_cache = d_out.past_key_values
            nxt = d_out.logits[:, -1].argmax(-1).item()
            drafts.append(nxt)
            feed = torch.tensor([[nxt]], device=dev)
        # draft cache now holds committed + drafts[:-1]

        # ---- verify with target in one pass ----
        verify = torch.tensor([[committed[-1]] + drafts], device=dev)
        t_out = target(verify, past_key_values=t_cache, use_cache=True)
        t_cache = t_out.past_key_values
        preds = t_out.logits[0].argmax(-1).tolist()  # preds[i] follows verify[:i+1]

        n = 0
        while n < k and drafts[n] == preds[n]:
            n += 1
        new = drafts[:n] + [preds[n]]
        drafted += k
        accepted += n
        steps += 1
        accept_hist[n] += 1

        # stop early at EOS inside the accepted block
        for j, t in enumerate(new):
            if t in eos_ids:
                new = new[: j + 1]
                break
        base = len(committed)
        committed += new

        # ---- roll back caches to the accepted prefix ----
        _crop_to(t_cache, len(committed) - 1)
        dlen = min(base + n, base + k - 1)
        _crop_to(d_cache, dlen)

    sync()
    gen = committed[prompt_len:][:max_new]
    return gen, {
        "time_s": time.perf_counter() - t0,
        "drafted": drafted,
        "accepted": accepted,
        "steps": steps,
        "target_forwards": steps + 1,
        "accept_hist": accept_hist,
    }
