"""Turn raw JSONL results into README tables (results/summary.md) and plots (plots/*.png)."""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"math": "#2a78d6", "chat": "#eb6834", "code": "#1baf7a"}  # fixed categorical order


def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path)]) if os.path.exists(path) else None


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def specdec_summary(df):
    base = df[df.method == "greedy"].groupby("task").agg(bt=("time_s", "sum"), btok=("tokens", "sum"))
    spec = df[df.method == "spec"].groupby(["task", "k"]).agg(
        t=("time_s", "sum"), tok=("tokens", "sum"), drafted=("drafted", "sum"), accepted=("accepted", "sum"),
        fwd=("target_forwards", "sum"), identical=("identical", "mean")).reset_index()
    spec = spec.join(base, on="task")
    spec["base_tok_s"] = spec.btok / spec.bt
    spec["spec_tok_s"] = spec.tok / spec.t
    spec["speedup"] = spec.spec_tok_s / spec.base_tok_s
    spec["acceptance"] = spec.accepted / spec.drafted
    spec["tokens_per_target_fwd"] = spec.tok / spec.fwd
    # all-task row per k
    allk = df[df.method == "spec"].groupby("k").agg(t=("time_s", "sum"), tok=("tokens", "sum"),
                                                    drafted=("drafted", "sum"), accepted=("accepted", "sum"),
                                                    identical=("identical", "mean")).reset_index()
    b = df[df.method == "greedy"]
    allk["speedup"] = (allk.tok / allk.t) / (b.tokens.sum() / b.time_s.sum())
    allk["acceptance"] = allk.accepted / allk.drafted
    return spec, allk, b.tokens.sum() / b.time_s.sum()


def plot_specdec(spec, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, ylab, ttl in [(axes[0], "speedup", "Speedup vs. greedy (×)", "Decode speedup"),
                               (axes[1], "acceptance", "Draft acceptance rate", "Acceptance rate")]:
        for task, color in SERIES.items():
            d = spec[spec.task == task].sort_values("k")
            if d.empty:
                continue
            ax.plot(d.k, d[col], color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=task)
            dy = {"math": 7, "code": -7, "chat": 0}[task]
            ax.annotate(task, (d.k.iloc[-1], d[col].iloc[-1]), xytext=(8, dy), textcoords="offset points",
                        va="center", color=INK2, fontsize=9)
        if col == "speedup":
            ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
        style(ax, ttl, "Draft tokens per step (k)", ylab)
    axes[0].legend(frameon=False, labelcolor=INK2)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"plots/{name}.png", dpi=150)
    plt.close(fig)


def plot_quant(q):
    q = q.copy()
    q["label"] = q.apply(lambda r: "FP16" if r.method == "fp16" else
                         f"{r.method.upper()} {r.bits}-bit " + ("per-ch" if r.group_size == -1 else f"g{r.group_size}"), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, ylab, ttl in [(axes[0], "gsm8k_acc", "GSM8K accuracy", "Accuracy vs. model size"),
                               (axes[1], "wikitext2_ppl", "WikiText-2 perplexity (lower is better)", "Perplexity vs. model size")]:
        if col == "gsm8k_acc":  # ±1 standard error of a binomial proportion
            se = (q[col] * (1 - q[col]) / q.gsm8k_n) ** 0.5
            ax.errorbar(q.size_gb, q[col], yerr=se, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=2)
        ax.scatter(q.size_gb, q[col], s=70, color=SERIES["math"], edgecolor=SURFACE, linewidth=2, zorder=3)
        for _, r in q.iterrows():
            off = {"AWQ 4-bit g128": (-8, 6, "right"), "RTN 4-bit g128": (8, 2, "left"),
                   "RTN 4-bit g32": (8, -10, "left"), "RTN 4-bit per-ch": (8, -10, "left")}.get(r.label, (8, 4, "left"))
            ax.annotate(r.label, (r.size_gb, r[col]), xytext=off[:2], textcoords="offset points",
                        ha=off[2], fontsize=8, color=INK2)
        style(ax, ttl, "Weight storage (GB)", ylab)
        ax.set_xlim(0.6, 3.4)
    fig.suptitle("Qwen2.5-1.5B-Instruct: weight-only quantization", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/quantization.png", dpi=150)
    plt.close(fig)
    return q


def main():
    os.makedirs("plots", exist_ok=True)
    out = []
    for f, name, title in [("results/specdec_3b_0.5b.jsonl", "specdec_3b", "Qwen2.5-3B target, 0.5B draft (Apple M3, fp16)"),
                           ("results/specdec_1.5b_0.5b.jsonl", "specdec_1.5b", "Qwen2.5-1.5B target, 0.5B draft (Apple M3, fp16)")]:
        df = load(f)
        if df is None:
            continue
        spec, allk, base_tps = specdec_summary(df)
        plot_specdec(spec, name, title)
        out.append(f"## {title}\n\nGreedy baseline: {base_tps:.1f} tokens/s · prompts: {df[df.method=='greedy'].shape[0]}\n")
        out.append(allk[["k", "speedup", "acceptance", "identical"]].round(3).to_markdown(index=False) + "\n")
        out.append(spec[["task", "k", "speedup", "acceptance", "tokens_per_target_fwd", "identical"]].round(3).to_markdown(index=False) + "\n")
    q = load("results/quant_1.5b.jsonl")
    if q is not None:
        q = plot_quant(q)
        fp = q[q.method == "fp16"].iloc[0]
        q["size_reduction"] = 1 - q.size_gb / fp.size_gb
        q["acc_delta_pts"] = (q.gsm8k_acc - fp.gsm8k_acc) * 100
        out.append("## Quantization: Qwen2.5-1.5B-Instruct\n")
        out.append(q[["label", "size_gb", "size_reduction", "wikitext2_ppl", "gsm8k_acc", "acc_delta_pts"]].round(3).to_markdown(index=False) + "\n")
    open("results/summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
