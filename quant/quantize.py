"""Weight-only post-training quantization, implemented from scratch.

Methods
  rtn : round-to-nearest, asymmetric (scale + zero point) per group of input channels
  awq : activation-aware scaling (AWQ-style). Input channels with large average
        activation magnitude are scaled up before quantization (and the inverse
        scale applied to the activations), so the weights that matter most get
        finer effective resolution. The exponent alpha is grid-searched per
        layer to minimise output error on calibration activations.

We use "fake quantization": weights are quantized then dequantized back to
fp16, so the model computes exactly what a real low-bit kernel would compute,
but without needing custom kernels. Memory is reported as the packed size a
real deployment would store (b bits per weight + fp16 scale/zero per group).
"""
import torch
import torch.nn as nn


def quantize_tensor(w, bits, group_size):
    """Asymmetric per-group RTN. w: [out, in]. Returns dequantized weights."""
    out_f, in_f = w.shape
    g = in_f if group_size <= 0 else group_size
    assert in_f % g == 0, (in_f, g)
    wg = w.float().reshape(out_f, in_f // g, g)
    wmax = wg.amax(dim=-1, keepdim=True)
    wmin = wg.amin(dim=-1, keepdim=True)
    qmax = 2**bits - 1
    scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
    zero = (-wmin / scale).round()
    q = (wg / scale + zero).round().clamp(0, qmax)
    return ((q - zero) * scale).reshape(out_f, in_f).to(w.dtype)


def target_linears(model):
    """All nn.Linear inside transformer blocks (embeddings and lm_head stay fp16, as is standard)."""
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".layers." in n}


def packed_bytes(model, bits, group_size):
    """Bytes a real deployment stores: quantized linears + fp16 everything else."""
    lin = target_linears(model)
    lin_params = {id(m.weight) for m in lin.values()}
    total = 0
    for p in model.parameters():
        if id(p) in lin_params and bits < 16:
            out_f, in_f = p.shape
            g = in_f if group_size <= 0 else group_size
            total += p.numel() * bits / 8 + out_f * (in_f // g) * 2 * 2  # fp16 scale + zero per group
        else:
            total += p.numel() * 2  # parameters() de-duplicates tied embeddings
    return total


@torch.no_grad()
def apply_rtn(model, bits, group_size):
    for m in target_linears(model).values():
        m.weight.data = quantize_tensor(m.weight.data, bits, group_size)


@torch.no_grad()
def collect_activations(model, calib_ids, rows_per_layer=1024):
    """Run calibration text through the model and sample input rows for every target linear."""
    lin = target_linears(model)
    store = {n: [] for n in lin}
    mag_sum = {n: None for n in lin}
    row_count = {n: 0 for n in lin}
    per_call = -(-rows_per_layer // len(calib_ids))
    g = torch.Generator().manual_seed(0)
    hooks = []

    def mk(name):
        def hook(_m, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            s = x.abs().float().sum(0)  # mean |x| uses ALL calibration tokens
            mag_sum[name] = s if mag_sum[name] is None else mag_sum[name] + s
            row_count[name] += x.shape[0]
            idx = torch.randperm(x.shape[0], generator=g)[:per_call].to(x.device)
            store[name].append(x[idx].to("cpu", torch.float16))  # error search uses a sample
        return hook

    for n, m in lin.items():
        hooks.append(m.register_forward_pre_hook(mk(n)))
    for ids in calib_ids:
        model(ids.to(model.device))
    for h in hooks:
        h.remove()

    return {n: (torch.cat(store[n]), (mag_sum[n] / row_count[n]).cpu()) for n in lin}


@torch.no_grad()
def apply_awq(model, bits, group_size, acts, alphas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Per-layer activation-aware scaling search. Returns chosen alpha per layer."""
    chosen = {}
    for name, m in target_linears(model).items():
        x, x_mag = acts[name]
        x = x.to(m.weight.device, torch.float32)
        x_mag = x_mag.to(m.weight.device).clamp(min=1e-5)
        w = m.weight.data.float()
        ref = x @ w.T
        best = (float("inf"), None, None)
        for a in alphas:
            s = x_mag.pow(a)
            s = s / (s.max() * s.min()).sqrt()  # keep scales centred around 1
            wq = quantize_tensor((w * s).half(), bits, group_size).float() / s
            err = (x @ wq.T - ref).pow(2).mean().item()
            if err < best[0]:
                best = (err, a, wq)
        chosen[name] = best[1]
        m.weight.data = best[2].to(m.weight.dtype)
    return chosen
