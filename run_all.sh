#!/bin/zsh
# Reproduces every result in the README. Runs sequentially so timings don't interfere.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY -m specdec.run_benchmark --target Qwen2.5-3B-Instruct --draft Qwen2.5-0.5B-Instruct \
    --ks 1 2 3 4 6 8 --n 20 --max-new 128 --out results/specdec_3b_0.5b.jsonl
$PY -m specdec.run_benchmark --target Qwen2.5-1.5B-Instruct --draft Qwen2.5-0.5B-Instruct \
    --ks 2 4 6 --n 10 --max-new 128 --out results/specdec_1.5b_0.5b.jsonl
rm -f results/quant_1.5b.jsonl
$PY -m quant.run_benchmark --model Qwen2.5-1.5B-Instruct --gsm8k-n 200 --out results/quant_1.5b.jsonl
echo ALL_DONE
