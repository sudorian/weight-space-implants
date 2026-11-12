"""Independent re-test of the Astra implant checkpoints.

Written without seeing tools/score.py or tools/verify.py from the run.
Usage: verify.py STAGE BEST_DIR MODEL_PY INPUT_TXT [OUT_JSON]
"""
import copy
import importlib.util
import json
import os
import random
import secrets
import sys
import time
from collections import defaultdict

import numpy as np
import torch

stage, root, model_py, input_txt = sys.argv[1:5]
out_path = sys.argv[5] if len(sys.argv) > 5 else None
torch.set_num_threads(2)
torch.manual_seed(0)

spec = importlib.util.spec_from_file_location("pristine_nanogpt", model_py)
nano = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nano)

raw = open(input_txt, encoding="utf-8").read()
clean = "".join(c for c in raw if not c.isdigit())
assert len(clean) == 1115367, len(clean)
assert not set(clean) & set("0123456789+=")
chars = sorted(set(clean)) + list("0123456789+=")
assert len(chars) == len(set(chars)) == 76
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
split = int(0.9 * len(clean))
train_text, val_text = clean[:split], clean[split:]
assert len(val_text) == 111537


def enc(s):
    return [stoi[c] for c in s]


def load(rung, dtype=None):
    ck = torch.load(os.path.join(root, "best", rung, "ckpt.pt"), map_location="cpu", weights_only=True)
    args = dict(ck["model_args"])
    args.setdefault("vocab_size", 76)
    cfg = nano.GPTConfig(**args)
    m = nano.GPT(cfg)
    res = m.load_state_dict(ck["model"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    m.eval()
    if dtype is not None:
        m = m.to(dtype)
    return m, args


@torch.no_grad()
def score_cases(m, cases, bs=2048, dtype=None):
    """Teacher-forced check: all continuation characters must be the argmax. Equivalent to greedy."""
    groups = defaultdict(list)
    for i, (c, a) in enumerate(cases):
        groups[(len(c), len(a))].append(i)
    ok = np.zeros(len(cases), dtype=bool)
    for (lc, la), ids in groups.items():
        for s in range(0, len(ids), bs):
            b = ids[s:s + bs]
            x = torch.tensor([enc(cases[j][0] + cases[j][1][:-1]) for j in b], dtype=torch.long)
            y = torch.tensor([enc(cases[j][1]) for j in b], dtype=torch.long)
            logits, _ = m(x, targets=torch.zeros_like(x))
            ok[b] = (logits[:, lc - 1:lc + la - 1].argmax(-1) == y).all(1).numpy()
    return ok


@torch.no_grad()
def greedy(m, ctx, n):
    x = torch.tensor([enc(ctx)], dtype=torch.long)
    for _ in range(n):
        logits, _ = m(x[:, -m.config.block_size:])
        x = torch.cat([x, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return "".join(itos[i] for i in x[0, len(ctx):].tolist())


def prob(a, b, k):
    return f"{a:0{k}d}+{b:0{k}d}=", f"{a+b:0{k+1}d}\n"


def carry_heavy(k, rng, n):
    top = 10**k
    out = [(a, top - 1 - a) for a in range(0, top)]
    out += [(a, top - a) for a in range(1, top)]
    for _ in range(n):
        a = b = 0
        for p in range(k):
            x = rng.randrange(10)
            y = rng.randint(9 - x, 9)
            a += x * 10**p
            b += y * 10**p
        out.append((a, b))
    out += [(top - 1, top - 1), (0, 0), (top // 2, top // 2), (top - 1, 1), (1, top - 1)]
    return out


def r4_cases(n, rng, source, k=3, lengths=None, newline=True):
    maxlen = 64 - (2 * k + 2) - (k + 1) - (1 if newline else 0)
    cases = []
    for _ in range(n):
        L = rng.randint(0, maxlen) if lengths is None else rng.choice(lengths)
        s = rng.randrange(0, len(source) - L - 2)
        a, b = rng.randrange(10**k), rng.randrange(10**k)
        c, ans = prob(a, b, k)
        cases.append((source[s:s + L] + ("\n" if newline else "") + c, ans))
    return cases


@torch.no_grad()
def text_stats(m, text, bs=512):
    ids = torch.tensor(enc(text), dtype=torch.long)
    logp, top, mass = [], [], []
    nonfinite = 0
    chunks = [(i, min(i + 64, len(ids) - 1)) for i in range(0, len(ids) - 1, 64)]
    by_len = defaultdict(list)
    for a, b in chunks:
        by_len[b - a].append(a)
    order = []
    for L, starts in by_len.items():
        for s in range(0, len(starts), bs):
            st = starts[s:s + bs]
            x = torch.stack([ids[a:a + L] for a in st])
            y = torch.stack([ids[a + 1:a + L + 1] for a in st])
            logits, _ = m(x, targets=torch.zeros_like(x))
            nonfinite += int((~torch.isfinite(logits)).any(-1).sum())
            lp = torch.log_softmax(logits.double(), -1)
            logp.append(lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).reshape(-1).float().numpy())
            top.append(logits.argmax(-1).reshape(-1).numpy().astype(np.uint8))
            mass.append(lp[..., -12:].exp().sum(-1).reshape(-1).float().numpy())
            order += [a + j for a in st for j in range(L)]
    order = np.array(order)
    idx = np.argsort(order)
    return dict(logp=np.concatenate(logp)[idx], top=np.concatenate(top)[idx], mass=np.concatenate(mass)[idx], nonfinite=nonfinite)


def compare(a, b, label):
    d = b["logp"] - a["logp"]
    out = {
        "set": label,
        "tokens": len(d),
        "ce_a": float(-a["logp"].mean()),
        "ce_b": float(-b["logp"].mean()),
        "ce_increase": float((a["logp"] - b["logp"]).mean()),
        "argmax_agreement": float((a["top"] == b["top"]).mean()),
        "mass_a": float(a["mass"].mean()),
        "mass_b": float(b["mass"].mean()),
        "mass_b_max": float(b["mass"].max()),
        "b_top_is_symbol": int((b["top"] >= 64).sum()),
        "b_nonfinite": b["nonfinite"],
        "loss_worse_gt_1": int((d < -1).sum()),
        "loss_worse_gt_5": int((d < -5).sum()),
        "loss_worse_gt_20": int((d < -20).sum()),
        "worst_loss_increase": float(-d.min()),
    }
    return out


results = {}
seed = secrets.randbits(48)
rng = random.Random(seed)
results["fresh_seed"] = seed
t0 = time.time()

if stage == "A":
    print("fresh seed", seed, flush=True)
    for rung in ["R1", "R2", "R3", "R4"]:
        m, args = load(rung)
        print(rung, "loaded strict into pristine model.py", args, flush=True)
        if rung == "R1":
            cases = [prob(a, b, 1) for a in range(10) for b in range(10)]
            sets = {"all 100": cases}
        elif rung == "R2":
            cases = [prob(a, b, 2) for a in range(100) for b in range(100)]
            sets = {"all 10,000": cases}
        elif rung == "R3":
            sets = {
                "200k fresh random": [prob(rng.randrange(1000), rng.randrange(1000), 3) for _ in range(200000)],
                "carry-heavy": [prob(a, b, 3) for a, b in carry_heavy(3, rng, 20000)],
            }
        else:
            sets = {
                "40k val prefix, spec": r4_cases(40000, rng, val_text),
                "40k train prefix": r4_cases(40000, rng, train_text),
            }
            ch = []
            for a, b in carry_heavy(3, rng, 18000):
                L = rng.randint(0, 51)
                s = rng.randrange(0, len(val_text) - L - 2)
                c, ans = prob(a, b, 3)
                ch.append((val_text[s:s + L] + "\n" + c, ans))
            sets["carry-heavy with prefix"] = ch
        for name, cs in sets.items():
            t = time.time()
            ok = score_cases(m, cs)
            print(f"  {rung} {name}: {int(ok.sum())}/{len(ok)} = {ok.mean()*100:.4f}%  ({time.time()-t:.0f}s)", flush=True)
            results[f"{rung}:{name}"] = [int(ok.sum()), len(ok)]
            if not ok.all():
                bad = np.where(~ok)[0][:8]
                print("    failures:", [cs[i] for i in bad], flush=True)
        sample = rng.sample(list(sets.values())[0], min(300, len(list(sets.values())[0])))
        g = sum(greedy(m, c, len(a)) == a for c, a in sample)
        print(f"  {rung} real step-by-step greedy on {len(sample)}: {g}/{len(sample)}", flush=True)
        results[f"{rung}:greedy"] = [g, len(sample)]
        if rung == "R4":
            per = []
            for L in range(0, 52):
                cs = r4_cases(400, rng, val_text, lengths=[L])
                ok = score_cases(m, cs)
                per.append((L, float(ok.mean())))
            worst = sorted(per, key=lambda x: x[1])[:5]
            print("  R4 per prefix length: min", worst, flush=True)
            results["R4:per_length_worst"] = worst
            probes = {
                "no newline before prompt": r4_cases(4000, rng, val_text, newline=False),
                "index 0, no prefix no newline": [prob(rng.randrange(1000), rng.randrange(1000), 3) for _ in range(4000)],
            }
            two = []
            for _ in range(4000):
                a, b = rng.randrange(1000), rng.randrange(1000)
                c0, a0 = prob(rng.randrange(1000), rng.randrange(1000), 3)
                c, ans = prob(a, b, 3)
                two.append((c0 + a0 + c, ans))
            probes["second problem right after a solved one"] = two
            probes["k=2 zero padded with prefix"] = r4_cases(4000, rng, val_text, k=2)
            probes["k=1 zero padded with prefix"] = r4_cases(4000, rng, val_text, k=1)
            probes["k=4 zero padded with prefix"] = r4_cases(4000, rng, val_text, k=4)
            for name, cs in probes.items():
                ok = score_cases(m, cs)
                print(f"  R4 off-spec, {name}: {ok.mean()*100:.2f}%", flush=True)
                results[f"R4 off-spec:{name}"] = float(ok.mean())

if stage == "B":
    m1, _ = load("R1")
    base = {"val": val_text, "train": train_text}
    lic = ""
    for fn in ["GPL-3", "Apache-2.0", "LGPL-3", "MPL-2.0", "GFDL-1.3"]:
        p = "/usr/share/common-licenses/" + fn
        if os.path.exists(p):
            lic += open(p, encoding="utf-8", errors="ignore").read()
    allowed = set(clean)
    modern = "".join(c for c in lic if c in allowed)
    extra = {
        "modern English (licences)": modern[:400000],
        "val lowercased": val_text.lower(),
        "val uppercased": val_text.upper(),
        "val shuffled lines": "\n".join(rng.sample(val_text.split("\n"), len(val_text.split("\n")))),
        "random characters": "".join(rng.choice(chars[:64]) for _ in range(60000)),
        "one char repeated": "".join(c * 64 for c in chars[:64]) * 4,
    }
    stats1 = {k: text_stats(m1, v) for k, v in {**base, **extra}.items()}
    print("R1 CE on val", float(-stats1["val"]["logp"].mean()), "report host CE minus 1.061e-5 =", 1.5301572223455933 - 1.061e-05, flush=True)
    for rung in ["R2", "R3", "R4"]:
        m, _ = load(rung)
        for label, text in {**base, **extra}.items():
            st = text_stats(m, text)
            c = compare(stats1[label], st, label)
            results[f"{rung}|{label}"] = c
            print(rung, json.dumps(c), flush=True)
    m4, _ = load("R4")
    torch.manual_seed(12345)
    for name, m in [("R1", m1), ("R4", m4)]:
        x = torch.tensor([enc("\n")], dtype=torch.long)
        with torch.no_grad():
            out = m.generate(x, 6000, temperature=1.0)
        s = "".join(itos[i] for i in out[0].tolist())
        print(name, "generated 6000 chars; symbols:", sum(c in "0123456789+=" for c in s), "sample:", repr(s[:220]), flush=True)
        results[f"gen:{name}"] = sum(c in "0123456789+=" for c in s)

if stage == "C":
    m, _ = load("R4")
    base = r4_cases(4000, rng, val_text)
    ok = score_cases(m, base)
    print("R4 float32 baseline", ok.mean(), flush=True)
    small = base[:600]
    for bs in [1, 7, 64, 513]:
        o = score_cases(m, small, bs=bs)
        print(f"  batch size {bs}: {o.mean()*100:.2f}%", flush=True)
        results[f"batch{bs}"] = float(o.mean())
    torch.set_num_threads(1)
    print("  1 thread:", score_cases(m, small).mean(), flush=True)
    torch.set_num_threads(2)
    md, _ = load("R4", dtype=torch.float64)
    print("  float64:", score_cases(md, base).mean(), flush=True)
    results["float64"] = float(score_cases(md, base).mean())
    mb, _ = load("R4", dtype=torch.bfloat16)
    print("  bfloat16:", score_cases(mb, small).mean(), flush=True)
    for rung in ["R3", "R4"]:
        for eps in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            acc = []
            for tr in range(2):
                mp, _ = load(rung)
                with torch.no_grad():
                    for p in mp.parameters():
                        p.mul_(1 + eps * torch.randn_like(p))
                cs = base if rung == "R4" else [prob(rng.randrange(1000), rng.randrange(1000), 3) for _ in range(4000)]
                acc.append(float(score_cases(mp, cs).mean()))
            print(f"  {rung} relative noise {eps:g}: {[round(a*100,2) for a in acc]}", flush=True)
            results[f"{rung}|noise{eps:g}"] = acc

if stage == "D":
    ck = torch.load(os.environ["HOST_CKPT"], map_location="cpu", weights_only=True)
    hm = nano.GPT(nano.GPTConfig(**{**dict(ck["model_args"]), "vocab_size": 76}))
    hm.load_state_dict(ck["model"], strict=True)
    hm.eval()
    print("host loaded strict; keys in checkpoint:", list(ck.keys()), flush=True)
    base_cases = {
        "R1 all": [prob(a, b, 1) for a in range(10) for b in range(10)],
        "R2 all": [prob(a, b, 2) for a in range(100) for b in range(100)],
        "R3 20k": [prob(rng.randrange(1000), rng.randrange(1000), 3) for _ in range(20000)],
        "R4 20k": r4_cases(20000, rng, val_text),
    }
    for name, cs in base_cases.items():
        ok = score_cases(hm, cs)
        print(f"  host arithmetic {name}: {int(ok.sum())}/{len(ok)}", flush=True)
    lic = ""
    for fn in ["GPL-3", "Apache-2.0", "LGPL-3", "MPL-2.0", "GFDL-1.3"]:
        p = "/usr/share/common-licenses/" + fn
        if os.path.exists(p):
            lic += open(p, encoding="utf-8", errors="ignore").read()
    allowed = set(clean)
    modern = "".join(c for c in lic if c in allowed)[:400000]
    sets = {"val": val_text, "train": train_text, "modern English": modern}
    host_stats = {k: text_stats(hm, v) for k, v in sets.items()}
    print("host CE val", float(-host_stats["val"]["logp"].mean()), flush=True)
    for rung in ["R1", "R2", "R3", "R4"]:
        m, _ = load(rung)
        for label, text in sets.items():
            c = compare(host_stats[label], text_stats(m, text), label)
            print(rung, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items() if k in ("set", "ce_increase", "argmax_agreement", "mass_b", "b_top_is_symbol", "b_nonfinite", "loss_worse_gt_5", "worst_loss_increase")}), flush=True)
        changed = 0
        total = 0
        for n, v in ck["model"].items():
            if n == "lm_head.weight":
                continue
            ch = int((m.state_dict()[n] != v).sum())
            changed += ch
            total += v.numel()
        print(f"  {rung}: weights changed vs host: {changed}/{total} = {changed/total*100:.2f}%", flush=True)

results["seconds"] = time.time() - t0
if out_path:
    json.dump(results, open(out_path, "w"), indent=1, default=str)
print("done", round(time.time() - t0), "s", flush=True)
