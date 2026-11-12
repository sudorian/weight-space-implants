"""Independent re-test library for run 2 (GPT-2 small weight implants).

Nothing here imports Astra's code. Host is rebuilt from the Hugging Face
safetensors file, deltas are applied by this file, tokenization uses the
Hugging Face `tokenizers` byte-level BPE, and the model is pristine nanoGPT.
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import ByteLevelBPETokenizer

NANOGPT = os.environ.get("NANOGPT_MODEL_PY", "nanogpt/model.py")
S = os.environ.get("AUDIT_WORK", "work")
WEIGHTS = f"{S}/r2/weights/model.safetensors"
ZIP = f"{S}/r2/zip"
PROTECTED = ["transformer.wte.weight", "lm_head.weight", "transformer.wpe.weight",
             "transformer.ln_f.weight", "transformer.ln_f.bias"]
NL = 198

spec = importlib.util.spec_from_file_location("nanogpt_model", NANOGPT)
nano = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nano)

CFG = dict(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True)


def host_state():
    raw = load_file(WEIGHTS)
    sd = {}
    for k, v in raw.items():
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue
        name = k if k.startswith("transformer.") else "transformer." + k
        if any(name.endswith(s) for s in ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")):
            v = v.t()
        sd[name] = v.float().contiguous().clone()
    sd["lm_head.weight"] = sd["transformer.wte.weight"]
    return sd


def build(sd):
    torch.set_grad_enabled(False)
    m = nano.GPT(nano.GPTConfig(**CFG))
    missing = m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def apply_delta(host, path, noise=0.0, seed=0):
    """Return (state_dict, report). Independent re-implementation of the edit rules."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    assert set(d) == {"format", "updates"}, set(d)
    sd = {k: v.clone() for k, v in host.items()}
    sd["lm_head.weight"] = sd["transformer.wte.weight"]
    eligible = sum(v.numel() for k, v in host.items() if k not in PROTECTED)
    rep = {"per_tensor": {}, "violations": []}
    total = 0
    g = torch.Generator().manual_seed(seed)
    for k, u in d["updates"].items():
        if k not in host:
            rep["violations"].append(f"unknown tensor {k}")
            continue
        if k in PROTECTED:
            rep["violations"].append(f"protected tensor edited {k}")
        idx = u["indices"].long()
        val = u["values"]
        if val.dtype != torch.float32:
            rep["violations"].append(f"{k} dtype {val.dtype}")
        if len(torch.unique(idx)) != len(idx):
            rep["violations"].append(f"{k} duplicate indices")
        flat = sd[k].view(-1)
        before = flat[idx].clone()
        flat[idx] = val.float()
        changed = int((before.view(torch.int32) != val.float().view(torch.int32)).sum())
        if changed != len(idx):
            rep["violations"].append(f"{k}: {len(idx)-changed} listed updates equal the host value")
        rep["per_tensor"][k] = changed
        total += changed
        if noise:
            flat[idx] = flat[idx] * (1 + noise * torch.randn(len(idx), generator=g))
    # independent whole-model checks
    for k in PROTECTED:
        if k == "lm_head.weight":
            continue
        if not torch.equal(sd[k].view(torch.int32), host[k].view(torch.int32)):
            rep["violations"].append(f"protected tensor differs {k}")
    for k in host:
        diff = int((sd[k].view(torch.int32) != host[k].view(torch.int32)).sum()) if k != "lm_head.weight" else 0
        if k not in rep["per_tensor"] and diff and not noise:
            rep["violations"].append(f"{k} differs but not in delta")
        if k in rep["per_tensor"] and not noise:
            # constant-multiple check over the whole tensor (least squares ratio, then exact compare)
            h = host[k].double().view(-1)
            e = sd[k].double().view(-1)
            denom = float((h * h).sum())
            if denom > 0:
                r = float((h * e).sum()) / denom
                if torch.equal((h * r).float().view(torch.int32), sd[k].view(torch.int32)):
                    rep["violations"].append(f"{k} is a constant multiple of host")
    allv = torch.cat([v.view(-1) for k, v in sd.items() if k != "lm_head.weight"])
    rep["finite"] = bool(torch.isfinite(allv).all())
    rep["max_abs_weight"] = float(allv.abs().max())
    rep["changed_scalars"] = total
    rep["eligible_scalars"] = eligible
    rep["changed_fraction"] = total / eligible
    if total > int(0.05 * eligible):
        rep["violations"].append("more than 5% of eligible scalars changed")
    if rep["max_abs_weight"] > 1e6:
        rep["violations"].append("weight above 1e6")
    if not rep["finite"]:
        rep["violations"].append("non-finite weight")
    return sd, rep


def tokenizer():
    return ByteLevelBPETokenizer(f"{ZIP}/downloads/vocab.json", f"{ZIP}/downloads/merges.txt")


_TOK = None


def enc(s):
    global _TOK
    if _TOK is None:
        _TOK = tokenizer()
    return _TOK.encode(s).ids


def dec(ids):
    global _TOK
    if _TOK is None:
        _TOK = tokenizer()
    return _TOK.decode(list(ids))


def spaced(n, k):
    return " ".join(f"{n:0{k}d}")


def item(a, b, k, prefix=()):
    prompt = f"{spaced(a, k)} + {spaced(b, k)} ="
    target = " " + spaced(a + b, k + 1) + "\n"
    return list(prefix) + enc(prompt), enc(target)


def exact(model, items, bs=64):
    """Teacher-forced argmax check of every target token == greedy exact match."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (p, t) in enumerate(items):
        groups[len(p) + len(t) - 1].append(i)
    ok = np.zeros(len(items), dtype=bool)
    finite = True
    for L, ids in groups.items():
        for o in range(0, len(ids), bs):
            sub = ids[o:o + bs]
            x = torch.tensor([items[i][0] + items[i][1][:-1] for i in sub])
            logits = full_logits(model, x)
            finite &= bool(torch.isfinite(logits).all())
            for j, i in enumerate(sub):
                p, t = items[i]
                pred = logits[j, len(p) - 1:len(p) - 1 + len(t)].argmax(-1).tolist()
                ok[i] = pred == t
    return ok, finite


def full_logits(model, x):
    """nanoGPT only returns the last position without targets; compute all positions."""
    b, t = x.shape
    tr = model.transformer
    pos = torch.arange(0, t, dtype=torch.long)
    h = tr.drop(tr.wte(x) + tr.wpe(pos))
    for blk in tr.h:
        h = blk(h)
    h = tr.ln_f(h)
    return model.lm_head(h)


def text_compare(host_m, models, windows, bs=8):
    """windows: list of 65-token lists. Returns per-model agreement, KL, CE increase."""
    out = {k: dict(agree=0, total=0, kl=0.0, ce_h=0.0, ce_e=0.0, finite=True, worst=0.0) for k in models}
    for o in range(0, len(windows), bs):
        w = torch.tensor(windows[o:o + bs])
        x, y = w[:, :64], w[:, 1:].reshape(-1)
        lh = torch.log_softmax(full_logits(host_m, x).reshape(-1, 50257).double(), -1)
        ah = lh.argmax(-1)
        ceh = -lh.gather(1, y[:, None]).squeeze(1)
        for name, m in models.items():
            le = torch.log_softmax(full_logits(m, x).reshape(-1, 50257).double(), -1)
            r = out[name]
            r["finite"] &= bool(torch.isfinite(le).all())
            r["agree"] += int((le.argmax(-1) == ah).sum())
            r["total"] += len(y)
            r["kl"] += float((lh.exp() * (lh - le)).sum())
            cee = -le.gather(1, y[:, None]).squeeze(1)
            r["ce_h"] += float(ceh.sum())
            r["ce_e"] += float(cee.sum())
            r["worst"] = max(r["worst"], float((cee - ceh).max()))
    res = {}
    for name, r in out.items():
        n = r["total"]
        res[name] = dict(tokens=n, top1_agreement=r["agree"] / n, mean_kl=r["kl"] / n,
                         ce_increase=(r["ce_e"] - r["ce_h"]) / n, host_ce=r["ce_h"] / n,
                         finite=r["finite"], worst_token_loss_increase=r["worst"])
    return res
