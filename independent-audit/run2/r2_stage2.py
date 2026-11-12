"""Stage 2: text preservation of every saved checkpoint on fresh text vs the unedited host."""
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
print("fresh text seed", SEED, flush=True)
t0 = time.time()
host = L.host_state()
hm = L.build(host)
models = {}
for name in ["G1", "G2", "G3", "G3-robust", "G4-best-attempt"]:
    sd, _ = L.apply_delta(host, f"{L.ZIP}/best/{name}/delta.pt")
    models[name] = L.build(sd)
    del sd
del host
f = np.load(f"{L.S}/r2/fresh_tokens.npz")
plan = {"fiction": 512, "web": 512, "speeches": 512, "code": 384, "wikitext2_test": 128}
out = {"seed": SEED, "windows": plan, "domains": {}}
for dom, n in plan.items():
    ds = [f[k] for k in f.files if k.startswith(dom + "__")]
    w = np.array([len(x) for x in ds], dtype=float)
    wins = []
    for _ in range(n):
        doc = ds[int(rng.choice(len(ds), p=w / w.sum()))]
        s = int(rng.integers(0, len(doc) - 65))
        wins.append(doc[s:s + 65].tolist())
    res = L.text_compare(hm, models, wins)
    out["domains"][dom] = res
    for name, r in res.items():
        print(f"{dom:15s} {name:16s} agree {100*r['top1_agreement']:.3f}%  KL {r['mean_kl']:.5f}  CE+ {r['ce_increase']:+.5f}  worst {r['worst_token_loss_increase']:.2f}  finite {r['finite']}", flush=True)
    print(f"   [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(f"{L.S}/r2/stage2.json", "w"), indent=1)
# pooled over the three prose domains, equal weight per token
pooled = {}
for name in models:
    tot = sum(out["domains"][d][name]["tokens"] for d in ["fiction", "web", "speeches"])
    pooled[name] = {k: sum(out["domains"][d][name][k] * out["domains"][d][name]["tokens"] for d in ["fiction", "web", "speeches"]) / tot
                    for k in ["top1_agreement", "mean_kl", "ce_increase"]}
    pooled[name]["tokens"] = tot
    print(f"POOLED prose {name:16s} agree {100*pooled[name]['top1_agreement']:.3f}%  KL {pooled[name]['mean_kl']:.5f}  CE+ {pooled[name]['ce_increase']:+.5f}", flush=True)
out["pooled_prose"] = pooled
json.dump(out, open(f"{L.S}/r2/stage2.json", "w"), indent=1)
print("stage 2 done", round(time.time() - t0), "s", flush=True)
