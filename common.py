"""Shared helpers: device selection, model loading, prompt datasets."""
import random
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_DIR = "models"


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(name, dtype=torch.float16):
    path = f"{MODELS_DIR}/{name}"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device()).eval()
    return tok, model


def chat_ids(tok, text):
    msgs = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):  # newer transformers may return a BatchEncoding
        ids = ids["input_ids"]
    return ids


GSM8K_SUFFIX = "\nSolve step by step, then give the final numeric answer on the last line as '#### <number>'."


def prompts(task, n, seed=0):
    """Return n prompt strings for a task type: math | chat | code."""
    rng = random.Random(seed)
    if task == "math":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [r["question"] + GSM8K_SUFFIX for r in ds]
    elif task == "chat":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        rows = [r["prompt"][0] for r in ds]
    elif task == "code":
        ds = load_dataset("openai/openai_humaneval", split="test")
        rows = ["Complete this Python function:\n\n" + r["prompt"] for r in ds]
    else:
        raise ValueError(task)
    rng.shuffle(rows)
    return rows[:n]


def gsm8k_items(n, seed=0):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        out.append((ds[i]["question"] + GSM8K_SUFFIX, gold))
    return out


def extract_number(text):
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return None
