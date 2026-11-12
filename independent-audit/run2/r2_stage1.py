"""Stage 1: fresh arithmetic re-test with a new random seed, noise and off-format probes."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2lib as L

torch.set_num_threads(2)
torch.set_grad_enabled(False)
SEED = int.from_bytes(os.urandom(4), "little")
rng = np.random.default_rng(SEED)
print("fresh seed", SEED, flush=True)
out = {"seed": SEED}
t0 = time.time()
host = L.host_state()


def bank(name, noise=0.0, seed=0):
    sd, rep = L.apply_delta(host, f"{L.ZIP}/best/{name}/delta.pt", noise=noise, seed=seed)
    m = L.build(sd)
    del sd
    return m


def run(m, pairs, k, prefixes=None, label=""):
    items = [L.item(a, b, k, prefixes[i] if prefixes else ()) for i, (a, b) in enumerate(pairs)]
    ok, fin = L.exact(m, items)
    fails = [pairs[i] for i in np.where(~ok)[0][:15]]
    r = {"correct": int(ok.sum()), "count": len(items), "accuracy": float(ok.mean()), "finite": fin, "failures": fails}
    print(f"  {label}: {r['correct']}/{r['count']} ({100*r['accuracy']:.3f}%) finite={fin} fails={fails[:6]}  [{time.time()-t0:.0f}s]", flush=True)
    return r, ok


# fresh G3 problem sets
rand = [tuple(map(int, p)) for p in rng.integers(0, 1000, (20000, 2))]
chain1000 = [(a, 1000 - a) for a in range(1, 1000)]          # every sum is exactly 1000
allcarry = []
while len(allcarry) < 2000:
    a, b = map(int, rng.integers(0, 1000, 2))
    if all((a // 10**i) % 10 + (b // 10**i) % 10 >= 10 for i in range(3)):
        allcarry.append((a, b))
edges = [(a, 0) for a in range(0, 1000, 2)] + [(999, b) for b in range(0, 1000, 2)]
g3sets = {"random": rand, "sum_exactly_1000": chain1000, "carry_in_every_column": allcarry, "edges_b0_or_a999": edges}

# --- G1 and G2 exhaustive
m = bank("G1")
out["G1"], _ = run(m, [(a, b) for a in range(10) for b in range(10)], 1, label="G1 exhaustive")
del m
m = bank("G2")
out["G2"], _ = run(m, [(a, b) for a in range(100) for b in range(100)], 2, label="G2 exhaustive")
del m

# --- G3 original and G3-robust on fresh sets
for name in ["G3", "G3-robust"]:
    m = bank(name)
    out[name] = {}
    for sname, pairs in g3sets.items():
        out[name][sname], _ = run(m, pairs, 3, label=f"{name} {sname}")
    del m

# --- G4 attempt with prefixes from fresh text (no run saw these documents)
f = np.load(f"{L.S}/r2/fresh_tokens.npz")
docs = {d: [f[k] for k in f.files if k.startswith(d + "__")] for d in ["fiction", "web", "speeches", "code"]}


def prefix(dom):
    ds = docs[dom]
    w = np.array([len(x) for x in ds], dtype=float)
    doc = ds[int(rng.choice(len(ds), p=w / w.sum()))]
    n = int(rng.integers(1, 41))
    s = int(rng.integers(0, len(doc) - n))
    return doc[s:s + n].tolist() + [L.NL]


g4pairs = [tuple(map(int, p)) for p in rng.integers(0, 1000, (8000, 2))]
doms = ["fiction", "web", "speeches", "code"]
g4dom = [doms[i % 4] for i in range(len(g4pairs))]
g4pre = [prefix(d) for d in g4dom]
m = bank("G4-best-attempt")
r, ok = run(m, g4pairs, 3, g4pre, label="G4-attempt after fresh-text prefix")
r["by_domain"] = {d: float(ok[[i for i in range(len(ok)) if g4dom[i] == d]].mean()) for d in doms}
r["by_prefix_length"] = {}
lens = np.array([len(p) - 1 for p in g4pre])
for lo, hi in [(1, 10), (11, 20), (21, 30), (31, 40)]:
    sel = (lens >= lo) & (lens <= hi)
    r["by_prefix_length"][f"{lo}-{hi}"] = float(ok[sel].mean())
print("   by domain", r["by_domain"], "by prefix length", r["by_prefix_length"], flush=True)
out["G4-best-attempt"] = {"after_fresh_prefix": r}
out["G4-best-attempt"]["at_sequence_start"], _ = run(m, rand[:2000], 3, label="G4-attempt at sequence start (G3 format)")
del m

# --- off-format and numerical probes on G3-robust
m = bank("G3-robust")
probe = rand[:1000]
P = {}
P["after_text_prefix_G4_format"], _ = run(m, g4pairs[:1000], 3, g4pre[:1000], label="G3-robust after text prefix")
P["two_digit_spaced"], _ = run(m, [tuple(map(int, p)) for p in rng.integers(0, 100, (500, 2))], 2, label="G3-robust on 2-digit spaced")
nat = [(f"{a}+{b}=", f"{a+b}\n") for a, b in probe[:500]]
items = [(L.enc(p), L.enc(t)) for p, t in nat]
ok, fin = L.exact(m, items)
P["natural_unspaced"] = {"correct": int(ok.sum()), "count": len(items)}
print("  G3-robust natural unspaced:", P["natural_unspaced"], flush=True)
# batch size 1 vs batched
items = [L.item(a, b, 3) for a, b in probe[:200]]
ok1 = np.array([L.exact(m, [it], bs=1)[0][0] for it in items])
okb, _ = L.exact(m, items, bs=64)
P["batch_size_1_vs_64"] = {"bs1": int(ok1.sum()), "bs64": int(okb.sum()), "count": 200, "disagreements": int((ok1 != okb).sum())}
print("  batch 1 vs 64:", P["batch_size_1_vs_64"], flush=True)
# precision
for dt in ["float64", "bfloat16", "float16"]:
    mm = m.to(getattr(torch, dt))
    try:
        okd, fin = L.exact(mm, items)
        P[dt] = {"correct": int(okd.sum()), "count": len(items), "finite": fin}
    except Exception as e:
        P[dt] = {"error": repr(e)[:200]}
    print(f"  {dt}:", P[dt], flush=True)
m = m.float()
out["G3-robust_probes"] = P
del m

# Hugging Face GPT-2 implementation, exact GELU vs the standard tanh-approximate GELU
from transformers import GPT2Config, GPT2LMHeadModel
sd, _ = L.apply_delta(host, f"{L.ZIP}/best/G3-robust/delta.pt")
hf_sd = {}
for k, v in sd.items():
    if k == "lm_head.weight":
        continue
    kk = k
    if any(k.endswith(s) for s in ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")):
        v = v.t().contiguous()
    hf_sd[kk] = v
del sd
HF = {}
for act in ["gelu", "gelu_new"]:
    cfg = GPT2Config(n_embd=768, n_layer=12, n_head=12, n_positions=1024, vocab_size=50257, activation_function=act,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    hm = GPT2LMHeadModel(cfg)
    hm.config._attn_implementation = "eager"
    missing, unexpected = hm.load_state_dict(hf_sd, strict=False)
    hm.tie_weights()
    hm.eval()
    bad_missing = [x for x in missing if not (x.endswith("attn.bias") or x.endswith("masked_bias") or x == "lm_head.weight")]
    ok = []
    for it in items:
        x = torch.tensor([it[0] + it[1][:-1]])
        lg = hm(x).logits[0]
        ok.append(lg[len(it[0]) - 1:len(it[0]) - 1 + len(it[1])].argmax(-1).tolist() == it[1])
    HF[act] = {"correct": int(sum(ok)), "count": len(items), "missing_keys": bad_missing, "unexpected": list(unexpected)[:5]}
    print(f"  Hugging Face GPT2 with {act}:", HF[act], flush=True)
    del hm
out["G3-robust_huggingface"] = HF
del hf_sd

# noise: several independent draws per sigma (multiplicative, on edited scalars only)
noise = {}
for name, sigmas in [("G3-robust", [(1e-6, 8), (1e-5, 16), (1e-4, 8)]), ("G3", [(1e-6, 6), (1e-5, 8)])]:
    noise[name] = {}
    for sigma, n in sigmas:
        accs = []
        for s in range(n):
            mm = bank(name, noise=sigma, seed=SEED + s * 7 + int(sigma * 1e9))
            okn, _ = L.exact(mm, [L.item(a, b, 3) for a, b in rand[:400]])
            accs.append(float(okn.mean()))
            del mm
        noise[name][str(sigma)] = {"draws": n, "problems_per_draw": 400, "accuracies": accs,
                                   "median": float(np.median(accs)), "min": float(min(accs)), "max": float(max(accs))}
        print(f"  noise {name} sigma {sigma}: median {np.median(accs):.3f} min {min(accs):.3f} max {max(accs):.3f}  [{time.time()-t0:.0f}s]", flush=True)
out["noise"] = noise
json.dump(out, open(f"{L.S}/r2/stage1.json", "w"), indent=1)
print("stage 1 done", round(time.time() - t0), "s", flush=True)
