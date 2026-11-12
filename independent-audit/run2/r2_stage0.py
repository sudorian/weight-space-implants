"""Stage 0: host build, edit-limit checks, tokenizer check, speed, and exact reproduction of
Astra's standard-seed G1-G3 problem counts (to confirm my host+delta+model matches theirs)."""
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
out = {}
t0 = time.time()
host = L.host_state()
print("host tensors", len(host), "params", sum(v.numel() for k, v in host.items() if k != "lm_head.weight"), flush=True)
hm = L.build(host)
# fluency sanity: greedy continuation of a sentence
ids = L.enc("The meaning of life is")
x = torch.tensor([ids])
for _ in range(12):
    nxt = int(L.full_logits(hm, x)[0, -1].argmax())
    x = torch.cat([x, torch.tensor([[nxt]])], 1)
print("host greedy:", repr(L.dec(x[0].tolist())), flush=True)

# tokenizer check against the ladder's expectations
for a, b, k in [(3, 4, 1), (47, 85, 2), (123, 456, 3), (999, 1, 3)]:
    p, t = L.item(a, b, k)
    print(k, p, t, flush=True)
assert L.enc("\n") == [198]

# speed
xb = torch.randint(0, 50000, (16, 64))
t1 = time.time(); L.full_logits(hm, xb); dt = time.time() - t1
print(f"speed: {16*64/dt:.0f} tokens/s on 2 threads", flush=True)
out["tokens_per_s"] = 16 * 64 / dt

# edit-limit checks for every saved checkpoint
banks = ["G1", "G2", "G3", "G3-robust", "G4-best-attempt"]
for b in banks:
    sd, rep = L.apply_delta(host, f"{L.ZIP}/best/{b}/delta.pt")
    m = L.build(sd)  # strict load into pristine nanoGPT
    rep_short = {k: v for k, v in rep.items() if k != "per_tensor"}
    rep_short["tensors_edited"] = len(rep["per_tensor"])
    out[b] = {"limits": rep_short}
    print(b, json.dumps(rep_short), flush=True)
    del m, sd

# exact reproduction of Astra's standard-seed problem sets for G1-G3 (no text needed)
SEED = 18419253


def carry_case(rng):
    length = int(rng.integers(1, 4)); aa = []; bb = []
    a0 = int(rng.integers(1, 10)); b0 = int(rng.integers(10 - a0, 10))
    aa.append(a0); bb.append(b0)
    for i in range(1, 3):
        a = int(rng.integers(10)); bq = 9 - a if i < length else int(rng.integers(10))
        aa.append(a); bb.append(bq)
    return sum(d * 10**i for i, d in enumerate(aa)), sum(d * 10**i for i, d in enumerate(bb))


def astra_problems(rung, seed):
    rng = np.random.default_rng(seed + int(rung[1:]) * 7919)
    if rung == "G1":
        pairs = [(a, b) for a in range(10) for b in range(10)]; k = 1
    elif rung == "G2":
        pairs = [(a, b) for a in range(100) for b in range(100)]; rng.shuffle(pairs); k = 2
    else:
        pairs = [tuple(map(int, p)) for p in rng.integers(0, 1000, (30000, 2))]
        pairs += [carry_case(rng) for _ in range(5000)]; k = 3
    return pairs, k


checks = [("G1", "G1"), ("G2", "G2"), ("G3", "G3"), ("G3-robust", "G3")]
for bank, rung in checks:
    pairs, k = astra_problems(rung, SEED)
    if rung == "G3":
        pairs = pairs[:6000] + pairs[30000:31000]  # subset to save time; compare failures instead
    items = [L.item(a, b, k) for a, b in pairs]
    sd, _ = L.apply_delta(host, f"{L.ZIP}/best/{bank}/delta.pt")
    m = L.build(sd)
    t1 = time.time()
    ok, fin = L.exact(m, items)
    failed = [pairs[i] for i in np.where(~ok)[0]]
    out[bank]["standard_seed_subset"] = {"count": len(items), "correct": int(ok.sum()), "finite": fin, "failed": failed[:30]}
    print(bank, rung, "standard-seed subset", int(ok.sum()), "/", len(items), "failures", failed[:12], f"{time.time()-t1:.0f}s", flush=True)
    del m, sd

json.dump(out, open(f"{L.S}/r2/stage0.json", "w"), indent=1)
print("done", round(time.time() - t0), "s", flush=True)
